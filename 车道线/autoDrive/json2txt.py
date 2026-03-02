import json
import os


def convert_to_yolo_format(json_data, class_id=0):
    data = json.loads(json_data)
    shapes = data.get('shapes', [])
    image_width = data['imageWidth']
    image_height = data['imageHeight']

    yolo_labels = []

    for shape in shapes:
        points = shape['points']

        yolo_points = []
        for point in points:
            x, y = point
            norm_x = x / image_width
            norm_y = y / image_height
            yolo_points.append(f"{norm_x:.6f} {norm_y:.6f}")

        yolo_label = f"{class_id} " + " ".join(yolo_points)
        yolo_labels.append(yolo_label)

    return "\n".join(yolo_labels)


def process_json_files(json_folder, output_folder):
    # 获取文件夹中的所有JSON文件
    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]

    for json_file in json_files:
        json_path = os.path.join(json_folder, json_file)

        with open(json_path, 'r') as file:
            json_data = file.read()

        # 转换为 YOLO 格式
        yolo_output = convert_to_yolo_format(json_data)

        # 将转换后的 YOLO 标签写入对应的 txt 文件
        txt_filename = os.path.splitext(json_file)[0] + '.txt'
        txt_path = os.path.join(output_folder, txt_filename)

        with open(txt_path, 'w') as file:
            file.write(yolo_output)

        print(f"Converted {json_file} to {txt_filename}")


# 使用该函数批量处理文件
json_folder = r'C:\Users\Lenovo\Desktop\华清远见\智能分拣\YOLO\autoDrive\cardata'  # 替换为你的 json 文件夹路径
output_folder = r'C:\Users\Lenovo\Desktop\华清远见\智能分拣\YOLO\autoDrive\cardata'  # 替换为你希望输出 .txt 文件的文件夹路径

process_json_files(json_folder, output_folder)
