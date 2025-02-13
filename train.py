"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_data_loader
from models import create_model
from util.visualizer import Visualizer


def main():
    # 解析训练配置
    options = TrainOptions().parse()
    dataset = create_data_loader(options)  # 创建数据集
    dataset_length = len(dataset)  # 数据集大小
    print(f'Total number of training images: {dataset_length}')

    # 创建并初始化模型
    model = create_model(options)
    model.setup(options)
    visualizer = Visualizer(options)  # 创建视觉化工具
    iteration_count = 0  # 初始化总迭代次数

    # 遍历所有的训练周期
    for epoch in range(options.epoch_count, options.n_epochs + options.n_epochs_decay + 1):
        epoch_start_time = time.time()  # 记录每个epoch的开始时间
        data_loading_time = time.time()  # 记录数据加载的时间
        epoch_iter = 0  # 当前epoch的训练迭代次数
        visualizer.reset()  # 重置visualizer，确保每个epoch至少保存一次
        model.update_learning_rate()  # 更新学习率

        # 遍历数据集进行训练
        for i, data in enumerate(dataset):
            iteration_start_time = time.time()  # 当前迭代的开始时间

            if iteration_count % options.print_freq == 0:
                data_loading_duration = iteration_start_time - data_loading_time  # 计算数据加载时间

            iteration_count += options.batch_size
            epoch_iter += options.batch_size
            model.set_input(data)  # 传入数据并进行预处理
            model.optimize_parameters()  # 计算损失并优化参数

            # 定期显示和保存图像
            if iteration_count % options.display_freq == 0:
                save_images = iteration_count % options.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_images)

            # 打印当前损失并记录日志
            if iteration_count % options.print_freq == 0:
                current_losses = model.get_current_losses()
                computation_time = (time.time() - iteration_start_time) / options.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, current_losses, computation_time,
                                                data_loading_duration)

                if options.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_length, current_losses)

            # 定期保存模型
            if iteration_count % options.save_latest_freq == 0:
                print(f'Saving the latest model (epoch {epoch}, iteration {iteration_count})')
                save_suffix = f'iter_{iteration_count}' if options.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            data_loading_time = time.time()  # 更新数据加载时间

        # 定期保存整个epoch的模型
        if epoch % options.save_epoch_freq == 0:
            print(f'Saving the model at the end of epoch {epoch}, iterations {iteration_count}')
            model.save_networks('latest')
            model.save_networks(epoch)

        # 打印epoch结束时间
        print(
            f'End of epoch {epoch} / {options.n_epochs + options.n_epochs_decay} \t Time Taken: {int(time.time() - epoch_start_time)} sec')


if __name__ == '__main__':
    main()

