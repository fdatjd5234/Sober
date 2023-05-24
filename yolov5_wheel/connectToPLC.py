import datetime
import time
import snap7
import pymysql


class ConnPlc:
    def __init__(self):
        self.ip = "10.200.35.3"  # PLC的IP地址
        self.rock = 0
        self.slot = 2

    def connect(self):
        plc = snap7.client.Client()
        plc.connect(self.ip, self.rock, self.slot)  # 连接到PLC
        return plc

    def getByMb(self):
        plc = self.connect()
        data = plc.mb_read(668, 1)  # 读取PLC指定地址的数据
        data = snap7.util.get_bool(data, 0, 0)  # 将数据转换为布尔值
        return data


class ConnMysql:
    def __init__(self):
        self.host = "127.0.0.1"  # 数据库主机地址
        self.port = 3306  # 数据库端口号
        self.user = "root"  # 数据库用户名
        self.pwd = "123456"  # 数据库密码
        self.database = "plc"  # 数据库名称
        self.charset = "utf8"  # 数据库字符集

    def connMysql(self):
        conn = pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.pwd,
            database=self.database,
            port=self.port,
            charset=self.charset,
        )  # 建立数据库连接
        return conn

    def writeToMysql(self, con, single, date):
        cursor = con.cursor()
        try:
            sql = "UPDATE plcsingle SET plcsingle.inPosition=%d WHERE plcsingle.id=1" % single
            row_count = cursor.execute(sql)  # 执行SQL语句更新数据库
            print("影响行数: %d, 信号比变化成：%d, %s" % (row_count, single, date))
            con.commit()
        except Exception as e:
            print(e)
            con.rollback()

        cursor.close()


if __name__ == '__main__':
    plc = ConnPlc()
    mysql = ConnMysql()
    con = mysql.connMysql()
    lastValue = ""
    copyData = ""
    while True:
        try:
            now = datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d %H:%M:%S")
            data = plc.getByMb()
            if data:
                copyData = "1"
            else:
                copyData = "0"

            if copyData == lastValue:
                pass
            elif copyData == "0":
                mysql.writeToMysql(con, 0, now)
            else:
                mysql.writeToMysql(con, 1, now)
            lastValue = copyData
            time.sleep(2)
        except Exception as e:
            continue
    con.close()
