import pandas as pd
from neo4j import GraphDatabase
import os
import csv

# Neo4j数据库连接信息
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "123456"


# 读取CSV文件
def read_csv_file(file_path):
    try:
        # 使用正确的参数读取CSV
        df = pd.read_csv(
            file_path,
            delimiter=',',
            quotechar='"',
            quoting=csv.QUOTE_ALL
        )

        # 去除列名中的双引号
        df.columns = df.columns.str.replace('"', '')

        print(f"成功读取CSV文件，共 {len(df)} 行数据")
        print(f"列名: {list(df.columns)}")
        print(f"首行数据示例: {df.iloc[0].to_dict()}")
        return df
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return None


# 连接Neo4j数据库
def connect_to_neo4j():
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        # 验证连接并获取版本信息
        with driver.session() as session:
            result = session.run("CALL dbms.components() YIELD versions RETURN versions[0] as version")
            version = result.single()["version"]
            print(f"成功连接到Neo4j数据库 (版本: {version})")

        return driver
    except Exception as e:
        print(f"连接Neo4j数据库失败: {e}")
        return None


# 导入数据到Neo4j (兼容3.5.9版本)
def import_data_to_neo4j(driver, df):
    if driver is None or df is None:
        print("无法导入数据，驱动程序或数据框为空")
        return

    try:
        with driver.session() as session:
            # 创建约束 - 兼容3.5.9版本语法
            try:
                session.run("CREATE CONSTRAINT ON (u:User) ASSERT u.user_id IS UNIQUE")
                print("创建User约束成功")
            except Exception as e:
                if "ConstraintAlreadyExists" in str(e):
                    print("User约束已存在，跳过创建")
                else:
                    raise e

            try:
                session.run("CREATE CONSTRAINT ON (b:Book) ASSERT b.work_id IS UNIQUE")
                print("创建Book约束成功")
            except Exception as e:
                if "ConstraintAlreadyExists" in str(e):
                    print("Book约束已存在，跳过创建")
                else:
                    raise e

            try:
                session.run("CREATE CONSTRAINT ON (a:Author) ASSERT a.name IS UNIQUE")
                print("创建Author约束成功")
            except Exception as e:
                if "ConstraintAlreadyExists" in str(e):
                    print("Author约束已存在，跳过创建")
                else:
                    raise e

            try:
                session.run("CREATE CONSTRAINT ON (p:Publisher) ASSERT p.name IS UNIQUE")
                print("创建Publisher约束成功")
            except Exception as e:
                if "ConstraintAlreadyExists" in str(e):
                    print("Publisher约束已存在，跳过创建")
                else:
                    raise e

            # 导入数据 - 使用兼容3.5.9的语法
            for index, row in df.iterrows():
                user_id = row['user_id']
                work_id = row['work_id']
                title = row['title']
                author = row['author']
                publisher = row['publisher']
                cover_image = row['cover_image']
                rating = row['rating']

                # 创建用户节点
                session.run("MERGE (u:User {user_id: $user_id})", user_id=user_id)

                # 创建书籍节点并设置属性
                session.run("""
                    MERGE (b:Book {work_id: $work_id})
                    ON CREATE SET b.title = $title, 
                                  b.cover_image = $cover_image,
                                  b.rating = $rating
                """, work_id=work_id, title=title, cover_image=cover_image, rating=rating)

                # 创建作者节点
                session.run("MERGE (a:Author {name: $author})", author=author)

                # 创建出版社节点
                session.run("MERGE (p:Publisher {name: $publisher})", publisher=publisher)

                # 创建关系
                session.run("""
                    MATCH (u:User {user_id: $user_id}), (b:Book {work_id: $work_id})
                    MERGE (u)-[:RATED]->(b)
                """, user_id=user_id, work_id=work_id)

                session.run("""
                    MATCH (b:Book {work_id: $work_id}), (a:Author {name: $author})
                    MERGE (b)-[:WRITTEN_BY]->(a)
                """, work_id=work_id, author=author)

                session.run("""
                    MATCH (b:Book {work_id: $work_id}), (p:Publisher {name: $publisher})
                    MERGE (b)-[:PUBLISHED_BY]->(p)
                """, work_id=work_id, publisher=publisher)

                # 每100行打印一次进度
                if (index + 1) % 100 == 0:
                    print(f"已处理 {index + 1}/{len(df)} 行数据")

        print(f"数据导入成功，共导入 {len(df)} 行数据")
    except Exception as e:
        print(f"数据导入失败: {e}")


# 主函数
def main():
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(script_dir, "book.csv")

    print(f"尝试读取文件: {csv_file_path}")

    if not os.path.exists(csv_file_path):
        print(f"CSV文件不存在: {csv_file_path}")
        print("请确保文件名为 'book.csv' 并与脚本在同一目录")
        return

    df = read_csv_file(csv_file_path)
    driver = connect_to_neo4j()

    if df is not None and driver is not None:
        import_data_to_neo4j(driver, df)
    else:
        print("数据或数据库连接无效，终止导入")

    # 关闭数据库连接
    if driver:
        driver.close()
        print("Neo4j数据库连接已关闭")


if __name__ == "__main__":
    main()