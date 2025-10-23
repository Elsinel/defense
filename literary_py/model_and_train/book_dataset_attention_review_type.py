import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms, models
from PIL import Image
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
import joblib
from tqdm import tqdm


class BookDataset(Dataset):
    def __init__(self, data_df, image_dir, text_model_path='../../model_cache/bert-base-uncased',
                 image_size=224, precompute_features=False,
                 user_encoder=None, author_encoder=None, publisher_encoder=None):
        self.data = data_df
        self.image_dir = image_dir

        # 类别编码
        self.user_encoder = LabelEncoder()
        self.author_encoder = LabelEncoder()
        self.publisher_encoder = LabelEncoder()

        # 类别编码：优先使用传入的预训练编码器，否则新建LabelEncoder
        self.user_encoder = user_encoder if user_encoder is not None else LabelEncoder()
        self.author_encoder = author_encoder if author_encoder is not None else LabelEncoder()
        self.publisher_encoder = publisher_encoder if publisher_encoder is not None else LabelEncoder()

        # 处理user_id编码（如果传入了预训练编码器，就用transform；否则用fit_transform）
        if user_encoder is not None:
            # 对训练集中未出现的用户，分配一个新的索引（冷启动）
            self.data['user_id'] = self.data['user_id'].astype(str).apply(
                lambda x: self.user_encoder.transform([x])[0] if x in self.user_encoder.classes_ else len(
                    self.user_encoder.classes_)
            )
        else:
            # 没有传入编码器时，重新拟合
            self.data['user_id'] = self.user_encoder.fit_transform(data_df['user_id'].astype(str))
            # 保存编码器（仅在首次训练时保存）
            os.makedirs('encoders', exist_ok=True)
            joblib.dump(self.user_encoder, 'encoders/user_encoder.pkl')

        # 处理author_id编码（同上）
        if author_encoder is not None:
            self.data['author_id'] = self.data['author'].apply(
                lambda x: self.author_encoder.transform([x])[0] if x in self.author_encoder.classes_ else len(
                    self.author_encoder.classes_)
            )
        else:
            self.data['author_id'] = self.author_encoder.fit_transform(data_df['author'])
            joblib.dump(self.author_encoder, 'encoders/author_encoder.pkl')

        # 处理publisher_id编码（同上）
        if publisher_encoder is not None:
            self.data['publisher_id'] = self.data['publisher'].apply(
                lambda x: self.publisher_encoder.transform([x])[0] if x in self.publisher_encoder.classes_ else len(
                    self.publisher_encoder.classes_)
            )
        else:
            self.data['publisher_id'] = self.publisher_encoder.fit_transform(data_df['publisher'])
            joblib.dump(self.publisher_encoder, 'encoders/publisher_encoder.pkl')

        base_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录
        model_path = os.path.abspath(os.path.join(base_dir, text_model_path))

        # 2. 打印验证信息
        print(f"模型加载路径: {model_path}")
        print(f"目录存在: {os.path.exists(model_path)}")
        if os.path.exists(model_path):
            print(f"目录内容: {os.listdir(model_path)}")
        self.text_tokenizer = BertTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )
        self.text_encoder = BertModel.from_pretrained(
            model_path,
            local_files_only=True
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.text_encoder = self.text_encoder.to(self.device)

        # 图片处理
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 标签处理：评分>4的为1，否则为0
        self.data['label'] = (data_df['rating'] > 4).astype(int)

        # 特征预计算
        self.precomputed_features = {}
        if precompute_features:
            self._precompute_features()

    def _safe_build_text(self, row):
        """安全构建文本，处理可能的类型问题"""
        title = row['title'] if isinstance(row['title'], str) else ""
        author = row['author'] if isinstance(row['author'], str) else ""
        publisher = str(row['publisher']) if pd.notna(row['publisher']) else ""

        # 移除abstract，只保留标题、作者和出版社
        return f"{title} by {author}. Published by {publisher}."

    def _precompute_features(self):
        print("批量预计算文本特征 (GPU加速)...")
        batch_size = 128
        feat_dim = self.text_encoder.config.hidden_size

        # 准备GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.text_encoder = self.text_encoder.to(self.device)
        torch.set_grad_enabled(False)

        # 创建连续数组存储所有特征
        all_features = np.zeros((len(self.data), feat_dim), dtype=np.float32)

        for i in tqdm(range(0, len(self.data), batch_size), desc="预计算"):
            # 获取批量文本
            batch_texts = []
            indices = []
            for j in range(i, min(i + batch_size, len(self.data))):
                row = self.data.iloc[j]
                text = self._safe_build_text(row)
                batch_texts.append(text)
                indices.append(j)

            # 批量分词
            inputs = self.text_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=128
            )

            # 移动数据到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 批量推理
            outputs = self.text_encoder(**inputs)
            features = outputs.last_hidden_state[:, 0, :]  # 使用[CLS]标记

            # 转换为CPU上的numpy数组
            batch_features = features.cpu().numpy().astype(np.float32)

            # 保存到数组的对应位置
            for idx, feat in zip(indices, batch_features):
                all_features[idx] = feat

        torch.set_grad_enabled(True)

        # 保存连续数组
        np.save('precomputed_text_features.npy', all_features)
        print(f"完成! 保存了 {all_features.shape[0]} 个文本特征")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # 用户ID
        user_id = torch.tensor(row['user_id'])

        # 文本特征（使用预计算或实时计算）
        if os.path.exists('precomputed_text_features.npy'):
            if not hasattr(self, 'text_features'):
                self.text_features = np.load('precomputed_text_features.npy')
                self.text_features_len = len(self.text_features)
                # 检查索引是否超出预计算特征数组范围
            if idx < self.text_features_len:
                text_feat = torch.tensor(self.text_features[idx])
            else:
                # 索引超出范围，实时计算文本特征
                text = self._safe_build_text(row)
                inputs = self.text_tokenizer(
                    text,
                    return_tensors='pt',
                    padding='max_length',
                    truncation=True,
                    max_length=128
                )

                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    text_feat = self.text_encoder(**inputs).last_hidden_state[:, 0, :].squeeze().cpu()
        else:
            text = self._safe_build_text(row)
            inputs = self.text_tokenizer(
                text,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=128
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                text_feat = self.text_encoder(**inputs).last_hidden_state[:, 0, :].squeeze().cpu()

        # 图片特征
        image_path = os.path.join(self.image_dir, row['cover_image'])
        if os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert('RGB')
                image = self.image_transform(image)
            except:
                image = torch.zeros(3, 224, 224)
        else:
            image = torch.zeros(3, 224, 224)

        # 类别特征
        author_id = torch.tensor(row['author_id'])
        publisher_id = torch.tensor(row['publisher_id'])

        # 标签
        label = torch.tensor(row['label'], dtype=torch.float)

        return {
            'user': user_id,
            'text': text_feat,
            'image': image,
            'author': author_id,
            'publisher': publisher_id,
            'label': label
        }


class AdaptiveBookRecommender(torch.nn.Module):
    def __init__(self, num_users, num_authors, num_publishers,
                 text_dim=768, image_feat_dim=1000,
                 user_emb_dim=64, author_emb_dim=32, publisher_emb_dim=32,
                 hidden_dim=256, resnet_weights_path="\model_cache\\resnet18-f37072fd.pth",
                 use_text=True,
                 use_image=True,
                 use_attention=True
                 ):
        super().__init__()

        self.use_text = use_text
        self.use_image = use_image
        self.use_attention = use_attention
        if not use_text and not use_image:
            raise ValueError("至少需要使用一种模态（文本或图像）")

        # 用户嵌入
        self.user_embedding = torch.nn.Embedding(num_users, user_emb_dim)

        # 作者和出版社嵌入
        self.author_embedding = torch.nn.Embedding(num_authors, author_emb_dim)
        self.publisher_embedding = torch.nn.Embedding(num_publishers, publisher_emb_dim)

        self.image_encoder = None
        # 图像特征提取器（使用预训练的ResNet）
        if self.use_image:
            self.image_encoder = self._load_resnet_from_local(resnet_weights_path)
            self.image_encoder.fc = torch.nn.Sequential(
                torch.nn.Linear(self.image_encoder.fc.in_features, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(512, image_feat_dim)
            )

        # 文本特征适配层（BERT特征已经是768维）
        self.text_adapter = None
        if self.use_text:
            self.text_adapter = torch.nn.Linear(text_dim, text_dim)

        # 上下文融合模块 - 生成上下文向量
        self.context_fusion = torch.nn.Sequential(
            torch.nn.Linear(user_emb_dim + author_emb_dim + publisher_emb_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )

        # 注意力机制（仅当同时使用文本和图像且启用注意力时初始化）
        self.text_attention = None
        self.image_attention = None
        if self.use_text and self.use_image and self.use_attention:
            self.text_attention = torch.nn.Sequential(
                torch.nn.Linear(text_dim + hidden_dim, hidden_dim),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_dim, 1)
            )
            self.image_attention = torch.nn.Sequential(
                torch.nn.Linear(image_feat_dim + hidden_dim, hidden_dim),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_dim, 1)
            )

        # 预测层
        fused_dim = 0
        if self.use_text:
            fused_dim += text_dim
        if self.use_image:
            fused_dim += image_feat_dim
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(fused_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, 1)
        )

    def _load_resnet_from_local(self, weights_path):
        """从本地文件加载预训练的ResNet18模型"""
        # 创建基础模型（不带预训练权重）
        model = models.resnet18(weights=None)

        # 从本地文件加载权重
        try:
            state_dict = torch.load(weights_path)
            # 适配模型结构变化
            model.load_state_dict(state_dict)
            print(f"成功从本地加载ResNet18权重: {weights_path}")
        except Exception as e:
            print(f"加载本地ResNet18权重失败: {str(e)}")
            print("使用未预训练的ResNet18模型")

        return model

    def forward(self, user, text, image, author, publisher):
        # 用户嵌入 - 确保索引不越界
        user = torch.clamp(user, 0, self.user_embedding.num_embeddings - 1)
        user_emb = self.user_embedding(user).squeeze(1)

        text_feat = None
        if self.use_text:
            text_feat = self.text_adapter(text)
        else:
            text_feat = torch.zeros_like(text)  # text维度已固定为768

        image_feat = None
        if self.use_image:
            image_feat = self.image_encoder(image)
        else:
            image_feat = torch.zeros(text.shape[0], 1000).to(image.device)  # 图像维度固定为1000

        # 类别特征 - 确保索引不越界
        author = torch.clamp(author, 0, self.author_embedding.num_embeddings - 1)
        publisher = torch.clamp(publisher, 0, self.publisher_embedding.num_embeddings - 1)
        author_emb = self.author_embedding(author).squeeze(1)
        publisher_emb = self.publisher_embedding(publisher).squeeze(1)

        # 融合上下文信息（用户、作者、出版社）生成上下文向量
        context_input = torch.cat([user_emb, author_emb, publisher_emb], dim=1)
        context_vector = self.context_fusion(context_input)

        # 特征融合（分情况处理）
        fused_features = None
        weights = None  # 用于返回注意力权重（无注意力时为None）

        if self.use_text and self.use_image:
            # 情况1：同时使用文本和图像
            if self.use_attention:
                # 有注意力机制：计算权重并加权融合
                text_attention_input = torch.cat([text_feat, context_vector], dim=1)
                text_attention_score = self.text_attention(text_attention_input)

                image_attention_input = torch.cat([image_feat, context_vector], dim=1)
                image_attention_score = self.image_attention(image_attention_input)

                attention_scores = torch.cat([text_attention_score, image_attention_score], dim=1)
                weights = torch.softmax(attention_scores, dim=1)

                weighted_text = weights[:, 0].unsqueeze(1) * text_feat
                weighted_image = weights[:, 1].unsqueeze(1) * image_feat
                fused_features = torch.cat([weighted_text, weighted_image], dim=1)
            else:
                # 无注意力机制：直接拼接
                fused_features = torch.cat([text_feat, image_feat], dim=1)

        elif self.use_text:
            # 情况2：仅使用文本
            fused_features = text_feat

        elif self.use_image:
            # 情况3：仅使用图像
            fused_features = image_feat

        # 最终预测 - 添加sigmoid激活函数
        pred = self.predictor(fused_features)
        pred = torch.sigmoid(pred)
        return pred.squeeze(), weights


# 训练函数
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, optimizer_name='adam', scheduler_type=None, step_size=5, gamma=0.1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = torch.nn.BCELoss()
    # 优化器选择
    if optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    elif optimizer_name.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name.lower() == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.99)
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")

    scheduler = None
    if scheduler_type is not None:
        if scheduler_type.lower() == 'steplr':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type.lower() == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif scheduler_type.lower() == 'reducelronplateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=2)
        else:
            raise ValueError(f"不支持的学习率调度器: {scheduler_type}")

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch + 1}/{epochs}, Learning Rate: {current_lr:.6f}')

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} Train"):
            user = batch['user'].to(device)
            text = batch['text'].to(device)
            image = batch['image'].to(device)
            author = batch['author'].to(device)
            publisher = batch['publisher'].to(device)
            label = batch['label'].to(device)

            optimizer.zero_grad()
            output, _ = model(user, text, image, author, publisher)
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # 验证
        model.eval()
        val_loss = 0
        val_acc = 0

        with torch.no_grad():
            for batch in val_loader:
                user = batch['user'].to(device)
                text = batch['text'].to(device)
                image = batch['image'].to(device)
                author = batch['author'].to(device)
                publisher = batch['publisher'].to(device)
                label = batch['label'].to(device)

                output, _ = model(user, text, image, author, publisher)
                loss = criterion(output, label)

                val_loss += loss.item()

                # 计算准确率
                pred = (output > 0.5).float()
                val_acc += (pred == label).sum().item()

        val_loss /= len(val_loader)
        val_acc /= len(val_loader.dataset)

        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # 更新学习率调度器
        if scheduler is not None:
            if scheduler_type.lower() == 'reducelronplateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

            # 保存最佳模型
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     save_model(model)

    return model


# 保存模型
def save_model(model, path='\model_and_train\pytorch_model'):
    if not os.path.exists(path):
        os.makedirs(path)

    torch.save(model.state_dict(), os.path.join(path, 'model.pth'))
    print(f'模型已保存到 {path} 目录')


# 加载模型
def load_model(model, path='\model_and_train\\test\pytorch_model'):
    model.load_state_dict(torch.load(os.path.join(path, 'model.pth')))
    print(f'模型已从 {path} 目录加载')
    return model


# 主训练函数
def train_main(data_path='.\\bookinfo_train.csv',
               image_dir='downloaded_images',
               precompute_features=False,
               optimizer_name='adam',
               scheduler_type=None,
               lr=0.001,
               step_size=5,
               gamma=0.1,
               model_type='full'):
    # 加载数据
    df = pd.read_csv(data_path)
    print(f'成功加载数据，共 {len(df)} 条记录')

    # 创建数据集
    dataset = BookDataset(df, image_dir, precompute_features=precompute_features)

    # 分割训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 根据模型类型设置参数
    if model_type == 'full':
        use_text = True
        use_image = True
        use_attention = True
        save_path = 'model_full'  # 模型保存路径区分
    elif model_type == 'text_only':
        use_text = True
        use_image = False
        use_attention = False  # 单模态无需注意力
        save_path = 'model_text_only'
    elif model_type == 'image_only':
        use_text = False
        use_image = True
        use_attention = False  # 单模态无需注意力
        save_path = 'model_image_only'
    elif model_type == 'no_attention':
        use_text = True
        use_image = True
        use_attention = False
        save_path = 'model_no_attention'
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    # 创建模型
    num_users = len(dataset.user_encoder.classes_)
    num_authors = len(dataset.author_encoder.classes_)
    num_publishers = len(dataset.publisher_encoder.classes_)

    model = AdaptiveBookRecommender(num_users, num_authors, num_publishers, use_text=use_text, use_image=use_image, use_attention=use_attention)

    # 训练模型
    # model = train_model(model, train_loader, val_loader)
    # 训练模型
    model = train_model(model, train_loader, val_loader, epochs=10, lr=lr, optimizer_name=optimizer_name,
                        scheduler_type=scheduler_type, step_size=step_size, gamma=gamma)

    # 保存模型
    save_model(model,save_path)

    print('模型训练完成')
    return model, dataset


def incremental_train_model(model, new_data_df, image_dir, precompute_features=False):
    """增量训练模型"""
    # 加载现有数据集
    df = pd.read_csv('.\\bookinfo_train.csv')
    # 合并新数据
    df = pd.concat([df, new_data_df], ignore_index=True)

    # 创建新的数据集
    dataset = BookDataset(df, image_dir, precompute_features=precompute_features)

    # 分割训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 增量训练
    model = train_model(model, train_loader, val_loader, epochs=3, lr=0.01, optimizer_name='sgd',scheduler_type='steplr', step_size=3, gamma=0.5)
    # model = train_model(model, train_loader, val_loader, epochs=3)  # 增量训练epoch数较少

    # 保存模型
    save_model(model)

    return model, dataset


def predict_batch(model, dataset, user_ids, book_infos, batch_size=32):
    """批量预测用户点击概率"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    predictions = []
    all_text_weights = []  # 存储文本权重
    all_image_weights = []

    # 分批次处理
    for i in range(0, len(book_infos), batch_size):
        batch_book_infos = book_infos[i:i + batch_size]
        batch_user_ids = user_ids[i:i + batch_size]

        # 准备输入张量
        user_tensors = []
        text_tensors = []
        image_tensors = []
        author_tensors = []
        publisher_tensors = []

        for user_id, book_info in zip(batch_user_ids, batch_book_infos):
            # 用户编码
            try:
                user_encoded = dataset.user_encoder.transform([str(user_id)])[0]
            except ValueError:
                user_encoded = len(dataset.user_encoder.classes_)

            # 作者编码
            author = book_info.get('author', 'Unknown')
            try:
                author_encoded = dataset.author_encoder.transform([author])[0]
            except ValueError:
                author_encoded = len(dataset.author_encoder.classes_)

            # 出版社编码
            publisher = book_info.get('publisher', 'Unknown')
            try:
                publisher_encoded = dataset.publisher_encoder.transform([publisher])[0]
            except ValueError:
                publisher_encoded = len(dataset.publisher_encoder.classes_)

            # 文本特征
            title = book_info.get('title', 'Unknown')
            text = f"{title} by {author}. Published by {publisher}."
            inputs = dataset.text_tokenizer(
                text,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=128
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                text_feat = dataset.text_encoder(**inputs).last_hidden_state[:, 0, :].squeeze()

            # 图像特征
            cover_image = book_info.get('cover_image', 'missing.jpg')
            image_path = os.path.join(dataset.image_dir, cover_image)
            if os.path.exists(image_path):
                try:
                    image = Image.open(image_path).convert('RGB')
                    image = dataset.image_transform(image)
                except:
                    image = torch.zeros(3, 224, 224)
            else:
                image = torch.zeros(3, 224, 224)

            # 收集张量
            user_tensors.append(torch.tensor([user_encoded]))
            text_tensors.append(text_feat)
            image_tensors.append(image)
            author_tensors.append(torch.tensor([author_encoded]))
            publisher_tensors.append(torch.tensor([publisher_encoded]))

        # 批量预测
            with torch.no_grad():
                # 处理空批次的情况
                if len(user_tensors) == 0:
                    continue

                user_tensor = torch.stack(user_tensors).to(device).squeeze(1)
                text_tensor = torch.stack(text_tensors).to(device)
                image_tensor = torch.stack(image_tensors).to(device)
                author_tensor = torch.stack(author_tensors).to(device).squeeze(1)
                publisher_tensor = torch.stack(publisher_tensors).to(device).squeeze(1)

                batch_pred, weight = model(
                    user_tensor,
                    text_tensor,
                    image_tensor,
                    author_tensor,
                    publisher_tensor
                )

                weight_np = weight.cpu().numpy()
                # 2. 提取文本权重（每行第一个元素）和图像权重（每行第二个元素）
                text_weights = weight_np[:, 0].tolist()  # 文本权重列表
                image_weights = weight_np[:, 1].tolist()

                # 确保预测结果是可迭代的
                if batch_pred.dim() == 0:  # 标量
                    batch_pred = batch_pred.unsqueeze(0)  # 转换为单元素张量

                # 转换为列表
                batch_pred_list = batch_pred.cpu().numpy().flatten().tolist()

                predictions.extend(batch_pred_list)
                all_text_weights.extend(text_weights)
                all_image_weights.extend(image_weights)

    return {
        'predictions': predictions,
        'text_weights': all_text_weights,  # 每个样本的文本权重
        'image_weights': all_image_weights  # 每个样本的图像权重
    }


def predict_user_click(model, dataset, user_id, book_info):
    """预测用户是否会点击某本书"""
    # 准备预测数据
    # 用户ID编码
    try:
        user_encoded = dataset.user_encoder.transform([str(user_id)])[0]
    except ValueError:
        # 冷启动策略：新用户
        user_encoded = len(dataset.user_encoder.classes_)

    # 作者编码
    author = book_info.get('author', 'Unknown')
    try:
        author_encoded = dataset.author_encoder.transform([author])[0]
    except ValueError:
        author_encoded = len(dataset.author_encoder.classes_)

    # 出版社编码
    publisher = book_info.get('publisher', 'Unknown')
    try:
        publisher_encoded = dataset.publisher_encoder.transform([publisher])[0]
    except ValueError:
        publisher_encoded = len(dataset.publisher_encoder.classes_)

    # 文本特征
    title = book_info.get('title', 'Unknown')
    text = f"{title} by {author}. Published by {publisher}."
    inputs = dataset.text_tokenizer(
        text,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=128
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    dataset.text_encoder = dataset.text_encoder.to(device)

    with torch.no_grad():
        text_feat = dataset.text_encoder(**inputs).last_hidden_state[:, 0, :].squeeze()

    # 图像特征
    cover_image = book_info.get('cover_image', 'missing.jpg')
    image_path = os.path.join('downloaded_images', cover_image)
    if os.path.exists(image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            image = dataset.image_transform(image)
        except:
            image = torch.zeros(3, 224, 224)
    else:
        image = torch.zeros(3, 224, 224)

    # 构建输入张量
    user_tensor = torch.tensor([user_encoded]).to(device)
    author_tensor = torch.tensor([author_encoded]).to(device)
    publisher_tensor = torch.tensor([publisher_encoded]).to(device)
    text_tensor = text_feat.unsqueeze(0).to(device)
    image_tensor = image.unsqueeze(0).to(device)

    # 预测
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        prediction, _ = model(user_tensor, text_tensor, image_tensor, author_tensor, publisher_tensor)

    return float(prediction)


# 推荐函数
def recommend_books(model, dataset, user_id, top_n=10,diversity=0.2):
    """为用户推荐图书"""
    try:
        # 检查用户是否存在
        try:
            user_encoded = dataset.user_encoder.transform([str(user_id)])[0]
        except ValueError:
            # 冷启动推荐
            return cold_start_recommend(top_n)

        # 获取用户已交互的书籍标题（因为没有book_id）
        user_interacted_titles = dataset.data[dataset.data['user_id'] == user_encoded]['title'].unique()

        # 获取所有书籍（去重）
        all_books = dataset.data.drop_duplicates(subset=['title', 'author', 'publisher'])

        # 筛选用户未交互的书籍
        candidate_books = all_books[~all_books['title'].isin(user_interacted_titles)]

        if candidate_books.empty:
            return cold_start_recommend(top_n)

        # 准备批量预测
        book_infos = []
        for _, row in candidate_books.iterrows():
            book_infos.append({
                'title': row['title'],
                'author': row['author'],
                'publisher': row['publisher'],
                'cover_image': row['cover_image']
            })

        # 批量预测
        result = predict_batch(
            model, dataset,
            [user_id] * len(candidate_books),
            book_infos
        )

        scores = result['predictions']
        text_weights = result['text_weights']
        image_weights = result['image_weights']

        print(f"候选书籍数量: {len(candidate_books)}, 预测分数数量: {len(scores)}")

        # 创建推荐列表（使用书籍的唯一标识）
        recommendations = []
        for idx, (_, row) in enumerate(candidate_books.iterrows()):
            # 创建书籍的唯一标识符（使用标题+作者+出版社）
            book_id = f"{row['title']}_{row['author']}_{row['publisher']}".replace(' ', '_')

            recommendations.append({
                'book_id': book_id,  # 生成的唯一ID
                'title': row['title'],
                'author': row['author'],
                'publisher': row['publisher'],
                'cover_image': row['cover_image'],
                'score': scores[idx],
                'text_contribution': text_weights[idx],  # 文本特征贡献（0~1）
                'image_contribution': image_weights[idx]  # 图像特征贡献（0~1）
            })

        # 排序并返回TopN
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        # 引入多样性：不直接取top_n，而是按分数分组后随机选择
        if len(recommendations) > top_n * 3:  # 只有候选书籍足够多时才应用多样性
            # 将推荐分为高、中、低分组
            high_group = recommendations[:top_n]
            mid_group = recommendations[top_n:2 * top_n]
            low_group = recommendations[2 * top_n:3 * top_n]

            # 按比例选择
            final_recs = []
            final_recs.extend(high_group[:int(top_n * (1 - 2 * diversity))])
            final_recs.extend(np.random.choice(mid_group, int(top_n * diversity), replace=False))
            final_recs.extend(np.random.choice(low_group, int(top_n * diversity), replace=False))

            # 随机打乱
            np.random.shuffle(final_recs)
            return final_recs
        else:
            return recommendations[:top_n]

    except Exception as e:
        # 记录错误信息
        print(f"推荐书籍时出错: {str(e)}")
        return cold_start_recommend(top_n)


def recall(model, dataset, user_id, top_n=50):
    """召回用户可能感兴趣的书籍
    Args:
        model: 训练好的推荐模型
        dataset: BookDataset实例
        user_id: 用户ID
        top_n: 召回数量
    Returns:
        召回书籍列表
    """
    # 检查用户是否存在
    try:
        user_encoded = dataset.user_encoder.transform([str(user_id)])[0]
    except ValueError:
        # 用户不存在，返回冷启动推荐
        return cold_start_recommend(dataset, top_n)

    # 获取用户已交互的书籍
    user_interacted_books = dataset.data[dataset.data['user_id'] == user_encoded]['book_id'].unique()

    # 获取所有书籍
    all_books = dataset.data['book_id'].unique()

    # 筛选用户未交互的书籍
    candidate_books = [book for book in all_books if book not in user_interacted_books]

    if not candidate_books:
        return []

    # 对每本候选书籍进行预测
    predictions = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for book_id in candidate_books:
            # 获取书籍信息
            book_info = dataset.data[dataset.data['book_id'] == book_id].iloc[0]

            # 作者编码
            author = book_info['author']
            try:
                author_encoded = dataset.author_encoder.transform([author])[0]
            except ValueError:
                author_encoded = len(dataset.author_encoder.classes_)

            # 出版社编码
            publisher = book_info['publisher']
            try:
                publisher_encoded = dataset.publisher_encoder.transform([publisher])[0]
            except ValueError:
                publisher_encoded = len(dataset.publisher_encoder.classes_)

            # 文本特征
            text = f"{book_info['title']} by {author}. Published by {publisher}."
            inputs = dataset.text_tokenizer(
                text,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=128
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            text_feat = dataset.text_encoder(**inputs).last_hidden_state[:, 0, :].squeeze()

            # 图像特征
            cover_image = book_info['cover_image']
            image_path = os.path.join(dataset.image_dir, cover_image)
            if os.path.exists(image_path):
                try:
                    image = Image.open(image_path).convert('RGB')
                    image = dataset.image_transform(image)
                except:
                    image = torch.zeros(3, 224, 224)
            else:
                image = torch.zeros(3, 224, 224)
            image = image.unsqueeze(0).to(device)

            # 构建输入张量
            user_tensor = torch.tensor([user_encoded]).to(device)
            author_tensor = torch.tensor([author_encoded]).to(device)
            publisher_tensor = torch.tensor([publisher_encoded]).to(device)
            text_tensor = text_feat.unsqueeze(0).to(device)

            # 预测
            prediction, _ = model(user_tensor, text_tensor, image, author_tensor, publisher_tensor)

            predictions.append((book_id, float(prediction)))

    # 按预测分数排序并选择前N本
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_books = predictions[:top_n]

    # 获取书籍详情
    result = []
    for book_id, score in top_books:
        book_info = dataset.data[dataset.data['book_id'] == book_id].iloc[0]
        result.append({
            'book_id': book_id,
            'title': book_info['title'],
            'author': book_info['author'],
            'publisher': book_info['publisher'],
            'cover_image': book_info['cover_image'],
            'score': score
        })

    return result

def recommend(model, dataset, user_id, top_n=10):
    """为用户推荐图书
    Args:
        model: 训练好的推荐模型
        dataset: BookDataset实例
        user_id: 用户ID
        top_n: 推荐数量
    Returns:
        推荐书籍列表
    """
    # 首先召回候选书籍
    candidate_books = recall(model, dataset, user_id, top_n * 5)  # 召回更多书籍以供排序

    # 按分数排序并选择前N本
    candidate_books.sort(key=lambda x: x['score'], reverse=True)
    recommended = candidate_books[:top_n]

    return recommended

def cold_start_recommend(top_n=10):
    """冷启动推荐，基于书籍的流行度和平均评分"""
    # 加载数据
    df = pd.read_csv('.\\book_ratings.csv')

    # 计算书籍流行度 (基于评分数量和平均评分)
    book_popularity = df.groupby('title').agg({
        'rating': ['count', 'mean'],
        'author': 'first',
        'publisher': 'first',
        'cover_image': 'first'
    }).reset_index()

    # 重命名列
    book_popularity.columns = ['title', 'rating_count', 'avg_rating', 'author', 'publisher', 'cover_image']

    # 计算综合得分 (评分数量 * 平均评分)
    book_popularity['score'] = book_popularity['rating_count'] * book_popularity['avg_rating']

    # 排序并选择前N本
    top_books = book_popularity.sort_values('score', ascending=False).head(top_n*3)

    if len(top_books) <= top_n:
        selected_books = top_books
    else:
        # 从候选池中随机选择top_n本
        selected_books = top_books.sample(n=top_n, random_state=None)

    # 格式化结果
    recommendations = []
    for _, row in selected_books.iterrows():
        recommendations.append({
            'title': row['title'],
            'author': row['author'],
            'publisher': row['publisher'],
            'cover_image': row['cover_image'],
            'score': float(row['score']),
            'avg_rating': float(row['avg_rating']),
            'rating_count': int(row['rating_count'])
        })
    np.random.shuffle(recommendations)

    return recommendations


def cold_start_recommend1(top_n=10, file_path='../../bookinfo_train.csv'):
    """冷启动推荐，基于书籍的流行度和平均评分（优化版）"""
    try:
        # 加载数据，指定rating为数值型，处理可能的空值
        df = pd.read_csv(
            file_path,
            dtype={'rating': float},  # 确保评分是数值型
            na_values=['', ' ']  # 识别空字符串为NaN
        )

        # 1. 数据清洗：移除关键字段为空的行
        critical_columns = ['title', 'author', 'publisher', 'rating', 'user_id']
        df_clean = df.dropna(subset=critical_columns).copy()

        # 2. 去重：同一用户对同一本书的多次评分只保留一次
        # 用title+author+publisher唯一标识一本书，结合user_id去重
        df_unique = df_clean.drop_duplicates(
            subset=['user_id', 'title', 'author', 'publisher'],
            keep='last'  # 保留最后一次评分
        )

        # 3. 分组计算：基于"书的唯一标识"（title+author+publisher）
        book_metrics = df_unique.groupby(
            ['title', 'author', 'publisher', 'cover_image']  # 复合主键确保同书同组
        ).agg({
            'rating': ['count', 'mean']  # 计算评分数量和平均分
        }).reset_index()

        # 重命名列
        book_metrics.columns = [
            'title', 'author', 'publisher', 'cover_image',
            'rating_count', 'avg_rating'
        ]

        # 4. 优化评分计算：用对数处理rating_count，平衡数量和质量
        # 避免评分数量过大的书籍过度主导（+1避免log(0)）
        book_metrics['score'] = (
                np.log1p(book_metrics['rating_count']) *  # 对数转换
                book_metrics['avg_rating']
        )

        # 5. 排序并筛选top_n
        if len(book_metrics) < top_n:
            print(f"警告：有效书籍数量不足{top_n}本，返回全部{len(book_metrics)}本")

        top_books = book_metrics.sort_values('score', ascending=False).head(top_n)

        # 格式化结果
        recommendations = top_books.to_dict('records')
        # 转换数值类型为基础类型（避免numpy类型序列化问题）
        for rec in recommendations:
            rec['rating_count'] = int(rec['rating_count'])
            rec['avg_rating'] = float(rec['avg_rating'])
            rec['score'] = float(rec['score'])

        return recommendations

    except FileNotFoundError:
        raise ValueError(f"文件不存在：{file_path}，请检查路径是否正确")
    except Exception as e:
        raise RuntimeError(f"推荐计算失败：{str(e)}")


def precision_at_k(recommended, relevant, k=10):
    recommended = recommended[:k]
    return len(set(recommended) & set(relevant)) / min(k, len(recommended)) if recommended else 0.0


def recall_at_k(recommended, relevant, k=10):
    if not relevant:
        return 0.0
    recommended = recommended[:k]
    return len(set(recommended) & set(relevant)) / len(relevant)


def ndcg_at_k(recommended, relevant, k=10):
    if not recommended or not relevant:
        return 0.0
    recommended = recommended[:k]
    dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(recommended) if item in relevant)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_top_users(model, dataset, test_df, top_n_users=50, k=10):
    """带进度显示的评估：只评估前N个有正样本的用户"""
    # 1. 筛选有正样本的用户（带进度条）
    print("正在筛选有正样本的用户...")
    user_groups = list(test_df.groupby('user_id'))  # 转换为列表便于进度条
    valid_users = []

    # 为筛选用户添加进度条
    for user_id, user_data in tqdm(user_groups, desc="筛选有效用户", total=len(user_groups)):
        if len(user_data[user_data['rating'] > 4]) > 0:  # 有至少1本喜欢的书
            valid_users.append(user_id)

    # 2. 只取前N个有效用户
    eval_users = valid_users[:top_n_users]
    print(f"\n共筛选出 {len(valid_users)} 个有正样本的用户，将评估前 {len(eval_users)} 个用户...")

    # 3. 计算这部分用户的指标（带详细进度）
    precisions, recalls, ndcgs = [], [], []

    # 为评估用户添加进度条，显示当前用户ID
    for idx, user_id in enumerate(tqdm(eval_users, desc=f"评估用户（共{len(eval_users)}个）", total=len(eval_users))):
        try:
            # 显示当前处理的用户（每10个用户打印一次，避免刷屏）
            if idx % 10 == 0:
                tqdm.write(f"正在处理第 {idx + 1}/{len(eval_users)} 个用户（ID: {user_id}）")

            # 获取该用户喜欢的书籍（正样本）
            user_data = test_df[test_df['user_id'] == user_id]
            relevant_books = user_data[user_data['rating'] > 4]['title'].unique().tolist()

            # 生成推荐列表
            recommendations = recommend_books(model, dataset, user_id, top_n=k)
            recommended_books = [rec['title'] for rec in recommendations]

            # 计算指标
            precisions.append(precision_at_k(recommended_books, relevant_books, k))
            recalls.append(recall_at_k(recommended_books, relevant_books, k))
            ndcgs.append(ndcg_at_k(recommended_books, relevant_books, k))

        except Exception as e:
            tqdm.write(f"⚠️ 用户{user_id}处理失败: {str(e)}")  # 在进度条中显示错误
            continue

    # 返回平均指标
    return {
        f"Precision@{k}": np.mean(precisions) if precisions else 0.0,
        f"Recall@{k}": np.mean(recalls) if recalls else 0.0,
        f"NDCG@{k}": np.mean(ndcgs) if ndcgs else 0.0
    }


def main_minimal_evaluation(model_path, encoder_dir='\model_and_train\encoders',
                            data_path='.\\bookinfo_train.csv',
                            image_dir='downloaded_images'):
    """带进度显示的极简评估主函数"""
    # 1. 加载数据（带进度提示）
    print("1/5 正在加载数据...")
    df = pd.read_csv(data_path)
    print(f"数据加载完成，共 {len(df)} 条记录")

    # 2. 抽取测试集（带进度提示）
    print("\n2/5 正在抽取测试集...")
    test_df = df.sample(frac=0.1, random_state=42)  # 只取10%数据作为测试集
    print(f"测试集抽取完成，共 {len(test_df)} 条记录")

    # 3. 加载编码器（带进度提示）
    print("\n3/5 正在加载编码器...")
    user_encoder = joblib.load(os.path.join(encoder_dir, 'user_encoder.pkl'))
    author_encoder = joblib.load(os.path.join(encoder_dir, 'author_encoder.pkl'))
    publisher_encoder = joblib.load(os.path.join(encoder_dir, 'publisher_encoder.pkl'))
    print(f"加载的编码器信息：")
    print(f"用户数量：{len(user_encoder.classes_)}")
    print(f"作者数量：{len(author_encoder.classes_)}")  # 此处应显示1275
    print(f"出版社数量：{len(publisher_encoder.classes_)}")
    print(f"编码器加载完成（用户数: {len(user_encoder.classes_)}, 作者数: {len(author_encoder.classes_)}）")

    # 4. 创建数据集和加载模型（带进度提示）
    print("\n4/5 正在初始化数据集和模型...")
    dataset = BookDataset(
        data_df=df,
        image_dir=image_dir,
        precompute_features=False,
        user_encoder=user_encoder,
        author_encoder=author_encoder,
        publisher_encoder=publisher_encoder
    )

    num_users = len(user_encoder.classes_)
    num_authors = len(author_encoder.classes_)
    num_publishers = len(publisher_encoder.classes_)
    model = AdaptiveBookRecommender(num_users, num_authors, num_publishers)
    model = load_model(model, path=model_path)
    print("模型加载完成")

    # 5. 评估模型（带详细进度条）
    print("\n5/5 开始评估...")
    metrics = evaluate_top_users(model, dataset, test_df, top_n_users=50, k=10)

    # 6. 输出结果
    print("\n===== 评估结果（前50个用户） =====")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    return metrics

if __name__ == '__main__':
    # train_main()
    # model_path = 'model_and_train\pytorch_model'
    #main_minimal_evaluation(model_path)
    # 使用SGD优化器和StepLR学习率调度
    # model, dataset = train_main(optimizer_name='sgd', scheduler_type='steplr', lr=0.01, step_size=3, gamma=0.5)

    # 1. 文本+图像+注意力（原模型）
    # model, dataset = train_main(
    #     optimizer_name='sgd',
    #     scheduler_type='steplr',
    #     lr=0.01,
    #     step_size=3,
    #     gamma=0.5,
    #     model_type='full'
    # )

    # 2. 仅文本
    # model, dataset = train_main(
    #     optimizer_name='sgd',
    #     scheduler_type='steplr',
    #     lr=0.01,
    #     step_size=3,
    #     gamma=0.5,
    #     model_type='text_only'
    # )

    # 3. 仅图像
    # model, dataset = train_main(
    #     optimizer_name='sgd',
    #     scheduler_type='steplr',
    #     lr=0.01,
    #     step_size=3,
    #     gamma=0.5,
    #     model_type='image_only'
    # )

    # 4. 文本+图像（无注意力）
    model, dataset = train_main(
        optimizer_name='sgd',
        scheduler_type='steplr',
        lr=0.01,
        step_size=3,
        gamma=0.5,
        model_type='no_attention'
    )

    # 使用AdamW优化器和CosineAnnealingLR学习率调度
    # model, dataset = train_main(optimizer_name='adamw', scheduler_type='cosine', lr=0.001)

    # 使用RMSprop优化器和ReduceLROnPlateau学习率调度
    # model, dataset = train_main(optimizer_name='rmsprop', scheduler_type='reducelronplateau', lr=0.005, gamma=0.1)
    # print(cold_start_recommend1())

    # 加载数据
    # data_path='../../bookinfo_info.csv'
    # df = pd.read_csv(data_path)
    # image_dir='downloaded_images'
    # print(f'成功加载数据，共 {len(df)} 条记录')
    #
    # # 创建数据集
    # dataset = BookDataset(df, image_dir, precompute_features=False)
    #
    # recommend_books()