import json
import os


def split_json(input_file, output_prefix, chunk_size, output_dir):
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    # Calculate total number of chunks
    total_chunks = len(data) // chunk_size
    if len(data) % chunk_size != 0:
        total_chunks += 1

    # Split data into chunks
    chunks = []
    for i in range(total_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        chunks.append(data[start_idx:end_idx])

    # Write chunks to files
    for i, chunk in enumerate(chunks):
        file_name = f"{output_prefix}_{i + 1}.json"
        output_file = os.path.join(output_dir, file_name)
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(chunk, outfile, ensure_ascii=False, indent=4)
        print(f"Created {output_file} with {len(chunk)} key-value pairs.")

if __name__ == '__main__':

    # 示例调用
    input_file = 'data/WN18RR/test.txt.json'  # 大JSON文件的路径
    output_prefix = 'test'    # 输出小文件的前缀
    output_dir = "data/WN18RR/small_test" # 输出文件所在的文件夹
    chunk_size = 100               # 每个小文件包含的条目数

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    split_json(input_file, output_prefix, chunk_size, output_dir)
