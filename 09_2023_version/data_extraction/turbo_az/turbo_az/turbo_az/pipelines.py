# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import cx_Oracle


# class TurboAzPipeline:
#     def process_item(self, item, spider):
#         return item
#


# class OraclePipeline:
#     def __init__(self, oracle_user, oracle_password, oracle_dsn):
#         self.oracle_user = oracle_user
#         self.oracle_password = oracle_password
#         self.oracle_dsn = oracle_dsn
#
#     @classmethod
#     def from_crawler(cls, crawler):
#         return cls(
#             oracle_user=crawler.settings.get('ORACLE_USER'),
#             oracle_password=crawler.settings.get('ORACLE_PASSWORD'),
#             oracle_dsn=crawler.settings.get('ORACLE_DSN')
#         )
#
#     def open_spider(self, spider):
#         self.conn = cx_Oracle.connect(self.oracle_user, self.oracle_password, self.oracle_dsn)
#
#     def close_spider(self, spider):
#         self.conn.close()
#
#     def process_item(self, item, spider):
#         cursor = self.conn.cursor()
#         cursor.execute("INSERT INTO your_table (title, content) VALUES (:title, :content)", item)
#         self.conn.commit()
#         cursor.close()
#         return item
