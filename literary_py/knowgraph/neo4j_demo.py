from neo4j import GraphDatabase, exceptions
import pandas as pd
import csv
import os


class Neo4jBookKnowledgeGraph:
    def __init__(self, uri, user, password):
        """初始化Neo4j连接"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """关闭数据库连接"""
        self.driver.close()

    def clear_database(self):
        """清空数据库（用于测试）"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("数据库已清空")

    # 暂时注释掉约束创建，以避免语法错误
    # def create_entity_constraints(self):
    #     """创建实体唯一性约束"""
    #     with self.driver.session() as session:
    #         # 用户唯一约束
    #         session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE")
    #         # 图书唯一约束
    #         session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (b:Book) REQUIRE b.id IS UNIQUE")
    #         # 作者唯一约束
    #         session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE")
    #         # 出版社唯一约束
    #         session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Publisher) REQUIRE p.name IS UNIQUE")
    #         # 类型唯一约束
    #         session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE")
    #     print("实体约束创建完成")

    def create_user(self, user_id):
        """创建用户节点"""
        with self.driver.session() as session:
            session.run(
                "MERGE (u:User {id: $user_id})",
                user_id=user_id
            )

    def create_book(self, book_id, title, description, cover, total_rating):
        """创建图书节点"""
        with self.driver.session() as session:
            session.run(
                """
                MERGE (b:Book {id: $book_id})
                SET b.title = $title,
                    b.description = $description,
                    b.cover = $cover,
                    b.total_rating = $total_rating
                """,
                book_id=book_id,
                title=title,
                description=description,
                cover=cover,
                total_rating=total_rating
            )

    def create_author(self, author_name):
        """创建作者节点"""
        with self.driver.session() as session:
            session.run(
                "MERGE (a:Author {name: $author_name})",
                author_name=author_name
            )

    def create_publisher(self, publisher_name):
        """创建出版社节点"""
        with self.driver.session() as session:
            session.run(
                "MERGE (p:Publisher {name: $publisher_name})",
                publisher_name=publisher_name
            )

    def create_genre(self, genre_name):
        """创建图书类型节点"""
        with self.driver.session() as session:
            session.run(
                "MERGE (g:Genre {name: $genre_name})",
                genre_name=genre_name
            )

    def create_rating_relationship(self, user_id, book_id, rating):
        """创建用户对图书的评分关系"""
        with self.driver.session() as session:
            session.run(
                """
                MATCH (u:User {id: $user_id})
                MATCH (b:Book {id: $book_id})
                MERGE (u)-[r:RATED {score: $rating}]->(b)
                """,
                user_id=user_id,
                book_id=book_id,
                rating=rating
            )

    def create_writes_relationship(self, author_name, book_id):
        """创建作者与图书的写作关系"""
        with self.driver.session() as session:
            session.run(
                """
                MATCH (a:Author {name: $author_name})
                MATCH (b:Book {id: $book_id})
                MERGE (a)-[r:WRITES]->(b)
                """,
                author_name=author_name,
                book_id=book_id
            )

    def create_publishes_relationship(self, publisher_name, book_id):
        """创建出版社与图书的出版关系"""
        with self.driver.session() as session:
            session.run(
                """
                MATCH (p:Publisher {name: $publisher_name})
                MATCH (b:Book {id: $book_id})
                MERGE (p)-[r:PUBLISHES]->(b)
                """,
                publisher_name=publisher_name,
                book_id=book_id
            )

    def create_belongs_to_relationship(self, book_id, genre_name):
        """创建图书与类型的归属关系"""
        with self.driver.session() as session:
            session.run(
                """
                MATCH (b:Book {id: $book_id})
                MATCH (g:Genre {name: $genre_name})
                MERGE (b)-[r:BELONGS_TO]->(g)
                """,
                book_id=book_id,
                genre_name=genre_name
            )

    def load_from_csv(self, csv_file):
        """从CSV文件加载数据并构建知识图谱"""
        try:
            df = pd.read_csv(csv_file)
            print(f"成功加载CSV文件，共 {len(df)} 条记录")

            # 先创建所有实体
            print("开始创建实体...")

            # 创建用户
            for user_id in df['用户id'].unique():
                self.create_user(f"user_{user_id}")

            # 创建图书
            for _, row in df.drop_duplicates('图书id').iterrows():
                self.create_book(
                    f"book_{row['图书id']}",
                    row['题名'],
                    row['简介'],
                    row['封面'],
                    row['总评分']
                )

            # 创建作者
            for author in df['作者'].unique():
                self.create_author(author)

            # 创建出版社
            for publisher in df['出版社'].unique():
                self.create_publisher(publisher)

            # 创建类型
            for genre in df['类型'].unique():
                self.create_genre(genre)

            print("实体创建完成，开始创建关系...")

            # 创建关系
            for _, row in df.iterrows():
                user_id = f"user_{row['用户id']}"
                book_id = f"book_{row['图书id']}"

                # 用户评分关系
                self.create_rating_relationship(user_id, book_id, row['用户评分'])

                # 作者写作关系
                self.create_writes_relationship(row['作者'], book_id)

                # 出版社出版关系
                self.create_publishes_relationship(row['出版社'], book_id)

                # 图书属于类型关系
                self.create_belongs_to_relationship(book_id, row['类型'])

            print("关系创建完成")

            # 统计节点和关系数量
            with self.driver.session() as session:
                nodes_count = session.run("MATCH (n) RETURN count(n) as count").single()['count']
                relationships_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()['count']

            print(f"知识图谱构建完成。节点数: {nodes_count}, 关系数: {relationships_count}")
            return True

        except Exception as e:
            print(f"加载CSV文件时出错: {str(e)}")
            return False


def create_sample_csv(file_path):
    """创建示例CSV数据文件"""
    data = [
        ["用户id", "图书id", "题名", "作者", "简介", "封面", "总评分", "用户评分", "出版社", "类型"],
        [1, 101, "Python编程", "张三", "Python入门书籍", "cover1.jpg", 4.5, 5, "编程出版社", "编程"],
        [1, 102, "数据结构", "李四", "数据结构与算法", "cover2.jpg", 4.2, 4, "计算机出版社", "编程"],
        [2, 101, "Python编程", "张三", "Python入门书籍", "cover1.jpg", 4.5, 4, "编程出版社", "编程"],
        [2, 103, "机器学习", "王五", "机器学习基础", "cover3.jpg", 4.7, 5, "科技出版社", "人工智能"],
        [3, 102, "数据结构", "李四", "数据结构与算法", "cover2.jpg", 4.2, 5, "计算机出版社", "编程"],
        [3, 103, "机器学习", "王五", "机器学习基础", "cover3.jpg", 4.7, 4, "科技出版社", "人工智能"],
        [3, 104, "深度学习", "赵六", "深度学习入门", "cover4.jpg", 4.6, 5, "科技出版社", "人工智能"]
    ]

    try:
        with open(file_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(data)
        print(f"示例CSV文件已创建: {file_path}")
        return True
    except Exception as e:
        print(f"创建示例CSV文件时出错: {str(e)}")
        return False


if __name__ == "__main__":
    # Neo4j连接信息
    neo4j_uri = "bolt://localhost:7687"
    neo4j_username = "neo4j"
    neo4j_password = "123456"

    # 示例CSV文件路径
    csv_file = ".\\neo4j-demo.csv"

    try:
        # 创建知识图谱实例并连接到Neo4j
        kg = Neo4jBookKnowledgeGraph(neo4j_uri, neo4j_username, neo4j_password)

        # 为测试方便，先清空数据库（实际使用时可注释掉）
        # kg.clear_database()

        # 暂时注释掉约束创建，避免语法错误
        # kg.create_entity_constraints()

        # 如果示例文件不存在，则创建它
        if not os.path.exists(csv_file):
            create_sample_csv(csv_file)

        # 从CSV文件加载数据
        kg.load_from_csv(csv_file)

        # 关闭连接
        kg.close()
        print("程序执行完成，已关闭数据库连接")

    except exceptions.AuthError:
        print("认证失败，请检查用户名和密码是否正确")
    except exceptions.ServiceUnavailable:
        print("无法连接到Neo4j服务，请检查服务是否已启动且地址正确")
    except Exception as e:
        print(f"发生错误: {str(e)}")
