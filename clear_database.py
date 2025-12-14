import asyncio
import asyncpg
import os
from pathlib import Path
from dotenv import load_dotenv

# 1. 加载环境变量 (保持不变)
current_dir = Path(__file__).resolve().parent
load_dotenv(dotenv_path=current_dir / '.env')


async def reset_database():
    # 获取数据库配置
    pg_host = os.getenv("POSTGRES_HOST", "localhost")
    pg_port = os.getenv("POSTGRES_PORT", 5432)
    pg_user = os.getenv("POSTGRES_USER", "postgres")
    pg_password = os.getenv("POSTGRES_PASSWORD", "postgres")
    pg_db = os.getenv("POSTGRES_DB", "kgplatform_chidu")

    # 定义 SQL 初始化文件的路径
    init_sql_path = current_dir / 'init_open_graph.sql'

    if not init_sql_path.exists():
        print(f"❌ 错误: 找不到初始化文件 {init_sql_path}")
        return

    print(f"正在连接数据库: {pg_host}:{pg_port}/{pg_db} ...")

    conn = None
    try:
        # 建立连接
        conn = await asyncpg.connect(
            host=pg_host,
            port=pg_port,
            user=pg_user,
            password=pg_password,
            database=pg_db
        )

        print("连接成功！")
        print(f"正在读取 SQL 文件: {init_sql_path.name} ...")

        # 2. 读取 SQL 文件内容
        sql_script = init_sql_path.read_text(encoding='utf-8')

        print("正在执行数据库重置操作（Drop & Create）...")

        # 3. 执行 SQL 脚本
        # asyncpg.execute 可以执行包含多条语句的脚本
        await conn.execute(sql_script)

        print("✅ 数据库重置成功！")
        print("   旧表已删除，并已根据 init_open_graph.sql 重新创建了空表。")

    except Exception as e:
        print(f"❌ 操作失败: {e}")
    finally:
        if conn:
            await conn.close()


if __name__ == "__main__":
    # 交互确认
    print("=" * 50)
    print("⚠️  高危操作警告 ⚠️")
    print("此操作将完全 DROP (删除) 现有数据库表，并根据 SQL 文件重新创建空表。")
    print("所有现有数据将永久丢失！")
    print("=" * 50)

    confirm = input("是否确认重置数据库？(请输入 'yes' 确认): ")
    if confirm.strip().lower() == 'yes':
        asyncio.run(reset_database())
    else:
        print("操作已取消。")