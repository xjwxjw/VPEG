import datetime
import os
import time

import cv2
import numpy as np
import skvideo.io
import tensorflow as tf
from PIL import Image
from tensorflow.python.platform import gfile

def get_next_video_data(data_dir):
    filenames = gfile.Glob(os.path.join(data_dir, '*'))
    if not filenames:
        raise RuntimeError('No data files found.')

    k = 0
    for f in filenames:
        
        for serialized_example in tf.python_io.tf_record_iterator(f):
            example = tf.train.Example()
            example.ParseFromString(serialized_example)
            # print(example)        # To know what all features are present

            actions = np.empty((0, 4), dtype='float')
            endeffector_positions = np.empty((0, 3), dtype='float')
            frames_aux1 = []
            frames_main = []
            i = 0
            while True:
                action_name = str(i) + '/action'
                action_value = np.array(example.features.feature[action_name].float_list.value)
                if action_value.shape == (0,):      # End of frames/data
                    break
                actions = np.vstack((actions, action_value))

                endeffector_pos_name = str(i) + '/endeffector_pos'
                endeffector_pos_value = list(example.features.feature[endeffector_pos_name].float_list.value)
                endeffector_positions = np.vstack((endeffector_positions, endeffector_pos_value))

                aux1_image_name = str(i) + '/image_aux1/encoded'
                aux1_byte_str = example.features.feature[aux1_image_name].bytes_list.value[0]
                aux1_img = Image.frombytes('RGB', (64, 64), aux1_byte_str)
                aux1_arr = np.array(aux1_img.getdata()).reshape((aux1_img.size[1], aux1_img.size[0], 3))
                frames_aux1.append(aux1_arr.reshape(1, 64, 64, 3))

                main_image_name = str(i) + '/image_main/encoded'
                main_byte_str = example.features.feature[main_image_name].bytes_list.value[0]
                main_img = Image.frombytes('RGB', (64, 64), main_byte_str)
                main_arr = np.array(main_img.getdata()).reshape((main_img.size[1], main_img.size[0], 3))
                frames_main.append(main_arr.reshape(1, 64, 64, 3))
                i += 1

            np_frames_aux1 = np.concatenate(frames_aux1, axis=0)
            np_frames_main = np.concatenate(frames_main, axis=0)
            yield f, k, actions, endeffector_positions, np_frames_aux1, np_frames_main
            k = k + 1


def extract_data(data_dir, output_dir, frame_rate):
    """
    Extracts data in tfrecord format to gifs, frames and text files
    :param data_dir:
    :param output_dir:
    :param frame_rate:
    :return:
    """
    if os.path.exists(output_dir):
        if os.listdir(output_dir):
            raise RuntimeError('Directory not empty: {0}'.format(output_dir))
    else:
        os.makedirs(output_dir)

    seq_generator = get_next_video_data(data_dir)
    while True:
        try:
            _, k, actions, endeff_pos, aux1_frames, main_frames = next(seq_generator)
        except StopIteration:
            break
        video_out_dir = os.path.join(output_dir, '{0:05}'.format(k))
        os.makedirs(video_out_dir)

        # noinspection PyTypeChecker
        np.savetxt(os.path.join(video_out_dir, 'actions.csv'), actions, delimiter=',')
        # noinspection PyTypeChecker
        np.savetxt(os.path.join(video_out_dir, 'endeffector_positions.csv'), endeff_pos, delimiter=',')
        skvideo.io.vwrite(os.path.join(video_out_dir, 'aux1.gif'), aux1_frames, inputdict={'-r': str(frame_rate)})
        skvideo.io.vwrite(os.path.join(video_out_dir, 'main.gif'), main_frames, inputdict={'-r': str(frame_rate)})
        skvideo.io.vwrite(os.path.join(video_out_dir, 'aux1.mp4'), aux1_frames, inputdict={'-r': str(frame_rate)})
        skvideo.io.vwrite(os.path.join(video_out_dir, 'main.mp4'), main_frames, inputdict={'-r': str(frame_rate)})

        # Save frames
        aux1_folder_path = os.path.join(video_out_dir, 'aux1_frames')
        os.makedirs(aux1_folder_path)
        for i, frame in enumerate(aux1_frames):
            filepath = os.path.join(aux1_folder_path, 'frame_{0:03}.bmp'.format(i))
            cv2.imwrite(filepath, cv2.cvtColor(frame.astype('uint8'), cv2.COLOR_RGB2BGR))
        main_folder_path = os.path.join(video_out_dir, 'main_frames')
        os.makedirs(main_folder_path)
        for i, frame in enumerate(main_frames):
            filepath = os.path.join(main_folder_path, 'frame_{0:03}.bmp'.format(i))
            cv2.imwrite(filepath, cv2.cvtColor(frame.astype('uint8'), cv2.COLOR_RGB2BGR))
        print('Saved video: {0:05}'.format(k))


def main():
    data_dir = './softmotion30_44k/train'
    output_dir = './ExtractedData/train'
    frame_rate = 4
    extract_data(data_dir, output_dir, frame_rate)
    return


if __name__ == '__main__':
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    main()
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))