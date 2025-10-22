from py2neo import Graph, Node, Relationship
import pandas as pd
from sqlalchemy import create_engine
import json
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('knowledge_graph_builder.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# 配置数据库连接
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "123456"
MYSQL_CONN = "mysql+pymysql://root:123456@localhost/literary_works_recommendation_platform"

# 初始化Neo4j
graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
graph.run("MATCH (n) DETACH DELETE n")  # 清空现有数据

# 创建MySQL引擎
engine = create_engine(MYSQL_CONN)

# 创建索引
graph.run("CREATE INDEX ON :User(user_id)")
graph.run("CREATE INDEX ON :Work(work_id)")
graph.run("CREATE INDEX ON :Tag(tag_id)")
graph.run("CREATE INDEX ON :Collection(collection_id)")
# 数据处理函数
def convert_value(value):
    """转换特殊数据类型为Neo4j兼容格式"""
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime('%Y-%m-%d %H:%M:%S')
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return value


def batch_create_nodes(node_type, data, primary_key, batch_size=1000):
    """
    批量创建节点（分批次处理）

    参数:
        node_type: 节点类型（如'User', 'Work'）
        data: 包含节点数据的DataFrame
        primary_key: 主键列名
        batch_size: 每批处理的行数

    返回:
        字典: {主键值: 节点对象}
    """
    logger.info(f"开始创建 {node_type} 节点，共 {len(data)} 个")
    nodes = {}
    total_count = len(data)
    num_batches = (total_count + batch_size - 1) // batch_size
    processed_count = 0

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, total_count)
        batch = data.iloc[start:end]

        tx = graph.begin()
        batch_nodes = {}

        for idx, row in batch.iterrows():
            properties = {}
            for col, value in row.items():
                col_name = str(col)
                properties[col_name] = convert_value(value)

            try:
                node = Node(node_type, **properties)
                tx.create(node)
                batch_nodes[row[primary_key]] = node
            except Exception as e:
                logger.error(f"创建节点失败 (行 {idx}): {str(e)}")
                logger.debug(f"问题数据: {properties}")

        try:
            tx.commit()
            nodes.update(batch_nodes)
            processed_count += len(batch)
            logger.info(f"批次 {batch_idx + 1}/{num_batches}: 成功创建 {len(batch)} 个 {node_type} 节点")
        except Exception as e:
            logger.error(f"提交批次 {batch_idx + 1} 失败: {str(e)}")
            tx.rollback()

    logger.info(f"完成创建 {node_type} 节点: {processed_count}/{total_count} 成功")
    return nodes


def batch_create_relationships(rel_type, data, from_nodes, to_nodes, from_key, to_key, extra_props=None,
                               batch_size=1000):
    """
    批量创建关系（分批次处理）

    参数:
        rel_type: 关系类型（如'OWNS', 'REVIEWED'）
        data: 包含关系数据的DataFrame
        from_nodes: 起始节点字典 {主键值: 节点对象}
        to_nodes: 目标节点字典 {主键值: 节点对象}
        from_key: 起始节点ID列名
        to_key: 目标节点ID列名
        extra_props: 额外属性列名列表
        batch_size: 每批处理的行数
    """
    logger.info(f"开始创建 {rel_type} 关系，共 {len(data)} 条")
    total_count = len(data)
    num_batches = (total_count + batch_size - 1) // batch_size
    success_count = 0
    skipped_count = 0

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, total_count)
        batch = data.iloc[start:end]

        tx = graph.begin()
        batch_success = 0
        batch_skipped = 0

        for idx, row in batch.iterrows():
            from_id = row[from_key]
            to_id = row[to_key]

            # 检查节点是否存在
            if from_id not in from_nodes or to_id not in to_nodes:
                batch_skipped += 1
                continue

            # 准备关系属性
            props = {}
            if extra_props:
                for prop in extra_props:
                    prop_name = str(prop)
                    if prop_name in row:
                        props[prop_name] = convert_value(row[prop_name])

            try:
                rel = Relationship(from_nodes[from_id], rel_type, to_nodes[to_id], **props)
                tx.create(rel)
                batch_success += 1
            except Exception as e:
                logger.error(f"创建关系失败 (行 {idx}): {str(e)}")
                batch_skipped += 1

        try:
            tx.commit()
            success_count += batch_success
            skipped_count += batch_skipped
            logger.info(f"批次 {batch_idx + 1}/{num_batches}: 创建 {batch_success} 条关系, 跳过 {batch_skipped} 条")
        except Exception as e:
            logger.error(f"提交批次 {batch_idx + 1} 失败: {str(e)}")
            tx.rollback()
            skipped_count += len(batch)

    logger.info(f"完成创建 {rel_type} 关系: 成功 {success_count}, 跳过 {skipped_count}, 总数 {total_count}")


# 数据加载函数
def load_data():
    """从MySQL加载所需数据"""
    logger.info("开始加载数据...")

    data = {}

    try:
        logger.info("加载用户数据...")
        data["users"] = pd.read_sql_table("user", engine)
        logger.info(f"加载 {len(data['users'])} 条用户数据")

        logger.info("加载作品数据...")
        data["works"] = pd.read_sql_table("work", engine)
        logger.info(f"加载 {len(data['works'])} 条作品数据")

        logger.info("加载标签数据...")
        data["tags"] = pd.read_sql_table("tag", engine)
        logger.info(f"加载 {len(data['tags'])} 条标签数据")

        logger.info("加载收藏夹数据...")
        data["collections"] = pd.read_sql_table("collection", engine)
        logger.info(f"加载 {len(data['collections'])} 条收藏夹数据")

        logger.info("加载收藏记录数据...")
        data["collection_works"] = pd.read_sql_table("record_collection_work", engine)
        logger.info(f"加载 {len(data['collection_works'])} 条收藏记录")

        logger.info("加载标签记录数据...")
        data["tag_works"] = pd.read_sql_table("record_tag_work", engine)
        logger.info(f"加载 {len(data['tag_works'])} 条标签记录")

        logger.info("加载浏览历史数据...")
        data["histories"] = pd.read_sql_table("history_user_work", engine)
        logger.info(f"加载 {len(data['histories'])} 条浏览历史")

        logger.info("加载评论数据...")
        data["reviews"] = pd.read_sql_table("review_user_work", engine)
        logger.info(f"加载 {len(data['reviews'])} 条评论数据")

        # 确保所有列名都是字符串
        for key in data:
            data[key].columns = data[key].columns.astype(str)

        # 转换主键列为整数类型（如果可能）
        for table, id_col in [("users", "user_id"), ("works", "work_id"),
                              ("tags", "tag_id"), ("collections", "collection_id")]:
            if table in data and id_col in data[table].columns:
                data[table][id_col] = data[table][id_col].astype(int)

        logger.info("数据加载完成")
    except Exception as e:
        logger.error(f"加载数据失败: {str(e)}")
        raise

    return data


# 主处理函数
def build_knowledge_graph():
    """构建知识图谱主函数"""
    start_time = time.time()
    logger.info("开始构建知识图谱")

    try:
        # 清空现有图谱（生产环境慎用）
        logger.warning("正在清空现有图谱...")
        graph.run("MATCH (n) DETACH DELETE n")

        # 创建索引

        # 加载数据
        data = load_data()

        # 创建节点
        logger.info("创建节点...")
        user_nodes = batch_create_nodes("User", data["users"], "user_id", batch_size=1000)
        work_nodes = batch_create_nodes("Work", data["works"], "work_id", batch_size=1000)
        tag_nodes = batch_create_nodes("Tag", data["tags"], "tag_id", batch_size=1000)
        collection_nodes = batch_create_nodes("Collection", data["collections"], "collection_id", batch_size=1000)

        # 创建关系
        logger.info("创建关系...")
        # 用户-收藏夹关系
        batch_create_relationships(
            "OWNS",
            data["collections"],
            user_nodes,
            collection_nodes,
            "owner_id",
            "collection_id",
            batch_size=2000
        )

        # 收藏夹-作品关系
        batch_create_relationships(
            "CONTAINS",
            data["collection_works"],
            collection_nodes,
            work_nodes,
            "collection_id",
            "work_id",
            ["create_time", "update_time"],
            batch_size=2000
        )

        # 作品-标签关系
        batch_create_relationships(
            "TAGGED_WITH",
            data["tag_works"],
            work_nodes,
            tag_nodes,
            "work_id",
            "tag_id",
            ["create_time", "update_time"],
            batch_size=2000
        )

        # 用户-作品浏览关系
        batch_create_relationships(
            "HAS_HISTORY",
            data["histories"],
            user_nodes,
            work_nodes,
            "user_id",
            "work_id",
            ["visit_count", "create_time", "update_time"],
            batch_size=2000
        )

        # 用户-作品评论关系
        batch_create_relationships(
            "REVIEWED",
            data["reviews"],
            user_nodes,
            work_nodes,
            "user_id",
            "work_id",
            ["rating", "content", "create_time", "update_time"],
            batch_size=2000
        )

        elapsed_time = time.time() - start_time
        logger.info(f"知识图谱构建完成！总耗时: {elapsed_time:.2f}秒")
        return True
    except Exception as e:
        logger.error(f"构建知识图谱失败: {str(e)}")
        return False


if __name__ == "__main__":
    build_knowledge_graph()