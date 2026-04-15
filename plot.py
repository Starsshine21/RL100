import re
import matplotlib.pyplot as plt

def plot_loss(file_path):
    epochs = []
    train_losses = []
    val_losses = []

    # 正则表达式：匹配每个Epoch结束时的汇总行
    # 例子: [IL] Epoch 0/1000, Loss: 0.1099, Val Loss: 0.0552
    pattern = re.compile(r"\[IL\] Epoch (\d+)/\d+, Loss: ([\d.]+), Val Loss: ([\d.]+)")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            matches = pattern.findall(content)
            
            for m in matches:
                epochs.append(int(m[0]) + 1) # Epoch从1开始显示
                train_losses.append(float(m[1]))
                val_losses.append(float(m[2]))

        if not epochs:
            print("未发现匹配的Epoch数据，请检查日志文件内容。")
            return

        plt.figure(figsize=(10, 5))

        # 1. 把点连起来: linestyle='-'
        # 2. 每个点缩小: markersize=3
        plt.plot(epochs, train_losses, 
                 label='Training Loss', 
                 color='#1f77b4', 
                 marker='o', 
                 markersize=3, 
                 linestyle='-', 
                 linewidth=1.5)

        plt.plot(epochs, val_losses, 
                 label='Validation Loss', 
                 color='#ff7f0e', 
                 marker='s', 
                 markersize=3, 
                 linestyle='-', 
                 linewidth=1.5)

        # 图表基础设置
        plt.title('Loss Curve per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.legend()

        # 保存图片
        plt.tight_layout()
        plt.savefig('loss_plot.png', dpi=300)
        print(f"成功解析 {len(epochs)} 个 Epoch。图像已保存为 loss_plot.png")
        plt.show()

    except Exception as e:
        print(f"读取文件出错: {e}")

if __name__ == "__main__":
    # 请确保同目录下有该日志文件，或者修改为实际路径
    log_file = "train.log" 
    plot_loss(log_file)