import os
import subprocess

# 要下载的文件 URL 列表
tar_urls = [
    "https://hf-mirror.com/datasets/q-future/q-align-datasets/resolve/main/koniq.tar",
    "https://hf-mirror.com/datasets/q-future/q-align-datasets/resolve/main/spaq.tar",
    "https://hf-mirror.com/datasets/q-future/q-align-datasets/resolve/main/kadid10k.tar",
]
# 输出目录
output_path = 'datasets/images'
os.makedirs(output_path, exist_ok=True)  # 创建目标目录


def download_and_extract(url, output_dir):
    # 构造文件名
    file_name = os.path.basename(url)
    output_file = os.path.join(output_dir, file_name)

    # 下载文件
    try:
        print(f"正在下载 {url} 到 {output_file} ...")
        subprocess.run(["wget", "-O", output_file, url], check=True)
    except subprocess.CalledProcessError:
        print(f"下载失败: {url}")
        return

    # 解压文件
    if os.path.exists(output_file):
        print(f"解压缩文件: {output_file}")
        try:
            subprocess.run(["tar", "-xf", output_file, "-C", output_dir], check=True)
        except subprocess.CalledProcessError:
            print(f"解压失败: {output_file}")
    else:
        print(f"文件不存在: {output_file}")


# 遍历下载每个文件
for url in tar_urls:
    download_and_extract(url, output_path)
