from neo4j import GraphDatabase
import uuid


class KnowledgeGraphBuilder:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def add_book(self, title, authors, publisher, cover, readers_ratings, summary):
        with self.driver.session() as session:
            session.execute_write(
                self._create_book_and_relations,
                title, authors, publisher, cover, readers_ratings, summary
            )

    @staticmethod
    def _create_book_and_relations(tx, title, authors, publisher, cover, readers_ratings, summary):
        # 创建书籍节点
        tx.run("""
        MERGE (b:Book {title: $title})
        SET b.cover = $cover, b.summary = $summary
        """, title=title, cover=cover, summary=summary)

        # 创建作者关系
        for author in authors:
            tx.run("""
            MATCH (b:Book {title: $title})
            MERGE (a:Author {name: $author})
            MERGE (a)-[:AUTHORED]->(b)
            """, title=title, author=author)

        # 创建出版社关系
        tx.run("""
        MATCH (b:Book {title: $title})
        MERGE (p:Publisher {name: $publisher})
        MERGE (p)-[:PUBLISHED]->(b)
        """, title=title, publisher=publisher)

        # 创建读者评分关系
        for reader, rating in readers_ratings.items():
            tx.run("""
            MATCH (b:Book {title: $title})
            MERGE (r:Reader {id: $reader_id})
            SET r.name = $reader_name
            MERGE (r)-[s:RATED]->(b)
            SET s.score = $rating
            """, title=title,
                   reader_id=str(uuid.uuid4()),
                   reader_name=reader,
                   rating=rating)


# 示例数据
if __name__ == "__main__":
    kg = KnowledgeGraphBuilder("bolt://localhost:7687", "neo4j", "123456")

    book_data = {
        "title": "Python编程从入门到实践",
        "authors": ["Eric Matthes"],
        "publisher": "人民邮电出版社",
        "cover": "https://example.com/cover1.jpg",
        "readers_ratings": {
            "张三": 4.5,
            "李四": 5.0,
            "王五": 3.8
        },
        "summary": "本书通过项目式教学方式，帮助读者快速掌握Python编程基础..."
    }

    kg.add_book(**book_data)

    # 添加第二本书
    book_data2 = {
        "title": "算法导论",
        "authors": ["Thomas H. Cormen", "Charles E. Leiserson"],
        "publisher": "机械工业出版社",
        "cover": "https://example.com/cover2.jpg",
        "readers_ratings": {
            "赵六": 4.8,
            "钱七": 4.2
        },
        "summary": "全面介绍算法原理和应用的经典教材..."
    }

    kg.add_book(**book_data2)

    kg.close()