import os

def extract_labels_from_filename(filename):
    return filename.split('_')[-1].split('.')[0]

def generate_results_txt(results_dir, output_txt):
    with open(output_txt, 'w') as f:
        for subdir in os.listdir(results_dir):
            subdir_path = os.path.join(results_dir, subdir)
            if os.path.isdir(subdir_path):
                # 找到测试图像文件名
                test_image = None
                labels = []
                for file in os.listdir(subdir_path):
                    if '_test' in file:
                        test_image = file
                    else:
                        labels.append(extract_labels_from_filename(file))
                print(test_image, len(labels))
                if test_image:
                    base_name = test_image.strip().split('_test')[0] + '.' + test_image.strip().split('.')[-1]
                    labels_str = ' '.join(labels)
                    f.write(f"{base_name} {labels_str}\n")

# 使用示例
results_dir = 'sim_outputs'
output_txt = 'results1111.txt'
generate_results_txt(results_dir, output_txt)
print(f"Results written to {output_txt}")
