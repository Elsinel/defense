import pandas as pd
from neo4j import GraphDatabase
import logging
from mysql_connector.sql_handler import SQLHandler
from typing import Dict, List, Optional

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Neo4jHandler:
    def __init__(self, uri: str, username: str, password: str):
        """初始化Neo4j数据库连接
        Args:
            uri: Neo4j数据库URI
            username: Neo4j用户名
            password: Neo4j密码
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        self._connect()

    def _connect(self):
        """建立Neo4j数据库连接"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, auth=(self.username, self.password)
            )
            # 测试连接
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info(f"成功连接到Neo4j数据库: {self.uri}")
        except Exception as e:
            logger.error(f"Neo4j数据库连接失败: {str(e)}")
            raise

    def close(self):
        """关闭Neo4j数据库连接"""
        if self.driver is not None:
            self.driver.close()
            logger.info("Neo4j数据库连接已关闭")

    def create_book_node(self, book_info: Dict):
        """创建书籍节点
        Args:
            book_info: 书籍信息字典，包含book_id, title, author, publisher, cover_image等字段
        """
        try:
            with self.driver.session() as session:
                session.execute_write(
                    self._create_book_node_tx,
                    book_info
                )
            logger.info(f"成功创建书籍节点: {book_info.get('title')}")
        except Exception as e:
            logger.error(f"创建书籍节点失败: {str(e)}")
            raise

    @staticmethod
    def _create_book_node_tx(tx, book_info: Dict):
        """创建书籍节点的事务函数"""
        query = "MERGE (b:Book {book_id: $book_id}) SET b.title = $title, b.author = $author, b.publisher = $publisher, b.cover_image = $cover_image RETURN b"
        result = tx.run(query, **book_info)
        return result.single()

    def create_user_node(self, user_id: str):
        """创建用户节点
        Args:
            user_id: 用户ID
        """
        try:
            with self.driver.session() as session:
                session.execute_write(
                    self._create_user_node_tx,
                    user_id
                )
            logger.info(f"成功创建用户节点: {user_id}")
        except Exception as e:
            logger.error(f"创建用户节点失败: {str(e)}")
            raise

    @staticmethod
    def _create_user_node_tx(tx, user_id: str):
        """创建用户节点的事务函数"""
        query = "MERGE (u:User {user_id: $user_id}) RETURN u"
        result = tx.run(query, user_id=user_id)
        return result.single()

    def create_author_node(self, author: str):
        """创建作者节点
        Args:
            author: 作者名称
        """
        try:
            with self.driver.session() as session:
                session.execute_write(
                    self._create_author_node_tx,
                    author
                )
            logger.info(f"成功创建作者节点: {author}")
        except Exception as e:
            logger.error(f"创建作者节点失败: {str(e)}")
            raise

    @staticmethod
    def _create_author_node_tx(tx, author: str):
        """创建作者节点的事务函数"""
        query = "MERGE (a:Author {name: $author}) RETURN a"
        result = tx.run(query, author=author)
        return result.single()

    def create_publisher_node(self, publisher: str):
        """创建出版社节点
        Args:
            publisher: 出版社名称
        """
        try:
            with self.driver.session() as session:
                session.execute_write(
                    self._create_publisher_node_tx,
                    publisher
                )
            logger.info(f"成功创建出版社节点: {publisher}")
        except Exception as e:
            logger.error(f"创建出版社节点失败: {str(e)}")
            raise

    @staticmethod
    def _create_publisher_node_tx(tx, publisher: str):
        """创建出版社节点的事务函数"""
        query = "MERGE (p:Publisher {name: $publisher}) RETURN p"
        result = tx.run(query, publisher=publisher)
        return result.single()

    def create_wrote_relationship(self, author: str, book_id: str):
        """创建作者写书籍的关系
        Args:
            author: 作者名称
            book_id: 书籍ID
        """
        try:
            with self.driver.session() as session:
                session.execute_write(
                    self._create_wrote_relationship_tx,
                    author, book_id
                )
            logger.info(f"成功创建作者{author}写书籍{book_id}的关系")
        except Exception as e:
            logger.error(f"创建作者写书籍关系失败: {str(e)}")
            raise

    @staticmethod
    def _create_wrote_relationship_tx(tx, author: str, book_id: str):
        """创建作者写书籍关系的事务函数"""
        query = """
        MATCH (a:Author {name: $author}), (b:Book {book_id: $book_id})
        MERGE (a)-[:WROTE]->(b)
        RETURN a, b
        """
        result = tx.run(query, author=author, book_id=book_id)
        return result.single()

    def create_published_relationship(self, publisher: str, book_id: str):
        """创建出版社出版书籍的关系
        Args:
            publisher: 出版社名称
            book_id: 书籍ID
        """
        try:
            with self.driver.session() as session:
                session.execute_write(
                    self._create_published_relationship_tx,
                    publisher, book_id
                )
            logger.info(f"成功创建出版社{publisher}出版书籍{book_id}的关系")
        except Exception as e:
            logger.error(f"创建出版社出版书籍关系失败: {str(e)}")
            raise

    @staticmethod
    def _create_published_relationship_tx(tx, publisher: str, book_id: str):
        """创建出版社出版书籍关系的事务函数"""
        query = """
        MATCH (p:Publisher {name: $publisher}), (b:Book {book_id: $book_id})
        MERGE (p)-[:PUBLISHED]->(b)
        RETURN p, b
        """
        result = tx.run(query, publisher=publisher, book_id=book_id)
        return result.single()

    def create_rated_relationship(self, user_id: str, book_id: str, rating: float):
        """创建用户评分书籍的关系
        Args:
            user_id: 用户ID
            book_id: 书籍ID
            rating: 评分值
        """
        try:
            with self.driver.session() as session:
                session.execute_write(
                    self._create_rated_relationship_tx,
                    user_id, book_id, rating
                )
            logger.info(f"成功创建用户{user_id}评分书籍{book_id}的关系，评分: {rating}")
        except Exception as e:
            logger.error(f"创建用户评分书籍关系失败: {str(e)}")
            raise

    @staticmethod
    def _create_rated_relationship_tx(tx, user_id: str, book_id: str, rating: float):
        """创建用户评分书籍关系的事务函数"""
        query = """
        MATCH (u:User {user_id: $user_id}), (b:Book {book_id: $book_id})
        MERGE (u)-[:RATED {score: $rating}]->(b)
        RETURN u, b
        """
        result = tx.run(query, user_id=user_id, book_id=book_id, rating=rating)
        return result.single()

    def insert_book_info(self, book_info: Dict):
        """插入书籍完整信息到知识图谱
        Args:
            book_info: 书籍信息字典，包含book_id, title, author, publisher, cover_image等字段
        """
        try:
            # 创建书籍节点
            self.create_book_node(book_info)

            # 创建作者节点
            author = book_info.get('author')
            if author:
                self.create_author_node(author)
                # 创建作者-书籍关系
                self.create_wrote_relationship(author, book_info.get('book_id'))

            # 创建出版社节点
            publisher = book_info.get('publisher')
            if publisher:
                self.create_publisher_node(publisher)
                # 创建出版社-书籍关系
                self.create_published_relationship(publisher, book_info.get('book_id'))

            logger.info(f"成功插入书籍完整信息: {book_info.get('title')}")
        except Exception as e:
            logger.error(f"插入书籍完整信息失败: {str(e)}")
            raise

    def insert_user_rating(self, user_id: str, book_info: Dict, rating: float):
        """插入用户评分信息到知识图谱
        Args:
            user_id: 用户ID
            book_info: 书籍信息字典
            rating: 评分值
        """
        try:
            # 创建用户节点
            self.create_user_node(user_id)

            # 插入书籍信息
            self.insert_book_info(book_info)

            # 创建用户评分关系
            self.create_rated_relationship(user_id, book_info.get('book_id'), rating)

            logger.info(f"成功插入用户{user_id}对书籍{book_info.get('title')}的评分: {rating}")
        except Exception as e:
            logger.error(f"插入用户评分信息失败: {str(e)}")
            raise

    def batch_insert_training_data(self, sql_handler):
        """批量插入训练数据到知识图谱
        Args:
            sql_handler: SQLHandler实例，用于获取训练数据
        """
        try:
            # 获取训练数据
            training_data = sql_handler.get_training_data()
            logger.info(f"成功获取训练数据，共 {len(training_data)} 条记录")

            # 批量插入数据
            for index, row in training_data.iterrows():
                # 构造书籍信息字典
                book_info = {
                    'book_id': row['work_id'],
                    'title': row['title'],
                    'author': row['author'],
                    'publisher': row['publisher'],
                    'cover_image': row['cover_image']
                }

                # 插入用户评分关系
                self.insert_user_rating(
                    user_id=str(row['user_id']),
                    book_info=book_info,
                    rating=row['rating']
                )

                # 每插入100条记录打印一次进度
                if (index + 1) % 100 == 0:
                    logger.info(f"已插入 {index + 1} 条记录")

            logger.info(f"批量插入完成，共插入 {len(training_data)} 条记录")
        except Exception as e:
            logger.error(f"批量插入训练数据失败: {str(e)}")
            raise

# 使用示例
if __name__ == '__main__':
    # 替换为实际的Neo4j连接信息
    neo4j_uri = "bolt://localhost:7687"
    neo4j_username = "neo4j"
    neo4j_password = "123456"

    try:
        # 创建Neo4j处理器
        neo4j_handler = Neo4jHandler(neo4j_uri, neo4j_username, neo4j_password)

        # 示例1：插入单个用户评分关系
        # 书籍信息
        book_info = {
            'book_id': '1',
            'title': '示例书籍',
            'author': '示例作者',
            'publisher': '示例出版社',
            'cover_image': 'example.jpg'
        }

        # 插入书籍信息
        neo4j_handler.insert_book_info(book_info)

        # 插入用户评分
        neo4j_handler.insert_user_rating('user1', book_info, 4.5)

        # 示例2：批量插入训练数据
        # 创建SQLHandler实例
        sql_conn_str = "mysql+pymysql://root:123456@localhost/literary_works_recommendation_platform"
        sql_handler = SQLHandler(sql_conn_str)

        try:
            # 批量插入训练数据
            neo4j_handler.batch_insert_training_data(sql_handler)
        finally:
            sql_handler.close()

        print("数据插入成功")

    except Exception as e:
        print(f"数据插入失败: {str(e)}")
    finally:
        # 关闭连接
        if 'neo4j_handler' in locals():
            neo4j_handler.close()
