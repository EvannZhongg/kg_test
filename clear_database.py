import asyncio
import asyncpg
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
current_dir = Path(__file__).resolve().parent
load_dotenv(dotenv_path=current_dir / '.env')


async def clear_data():
    # 获取数据库配置
    pg_host = os.getenv("POSTGRES_HOST", "localhost")
    pg_port = os.getenv("POSTGRES_PORT", 5432)
    pg_user = os.getenv("POSTGRES_USER", "postgres")
    pg_password = os.getenv("POSTGRES_PASSWORD", "postgres")
    pg_db = os.getenv("POSTGRES_DB", "kgplatform_chidu")

    print(f"正在连接数据库: {pg_host}:{pg_port}/{pg_db} ...")

    try:
        # 建立连接
        conn = await asyncpg.connect(
            host=pg_host,
            port=pg_port,
            user=pg_user,
            password=pg_password,
            database=pg_db
        )

        print("连接成功！准备清空数据...")

        # 定义要清空的表（注意顺序，使用了 CASCADE 会自动处理依赖，但显式列出更清晰）
        # RESTART IDENTITY 会将 ID 计数器重置为 1
        sql = """
        TRUNCATE TABLE 
            open_graph_provenance,
            open_graph_edges,
            open_graph_nodes,
            open_graph_chunks
        RESTART IDENTITY CASCADE;
        """

        # 执行
        await conn.execute(sql)
        print("✅ 所有业务表数据已清空，ID计数器已重置。")
        print("   (open_graph_chunks, open_graph_nodes, open_graph_edges, open_graph_provenance)")

        await conn.close()

    except Exception as e:
        print(f"❌ 操作失败: {e}")


if __name__ == "__main__":
    # 简单的确认交互
    confirm = input("⚠️  警告: 此操作将永久删除数据库中的所有图谱数据！\n是否继续？(y/n): ")
    if confirm.lower() == 'y':
        asyncio.run(clear_data())
    else:
        print("操作已取消。")