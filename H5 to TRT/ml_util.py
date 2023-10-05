import numpy as np
import os
import pandas as pd

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import tensorflow.keras.backend as kb
import time
# import model_util
from tensorflow.keras.callbacks import CSVLogger
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class CustomError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def get_files_to_use(root_folder, subject_nums, sides, trial_nums):
    subject_strings = convert_subject_nums_to_strings(subject_nums)
    trial_strings = convert_trial_nums_to_strings(trial_nums)
    files_to_use = []
    for subject_string in subject_strings:
        subject_folder = os.path.join(root_folder, subject_string)
        filenames = os.listdir(subject_folder)
        for side in sides:
            for trial_string in trial_strings:
                for f in filenames:
                    if side in f and trial_string in f:
                        files_to_use.append(os.path.join(subject_folder, f))
    return(files_to_use)


def convert_trial_nums_to_strings(list_of_trial_nums: int):
    mystrings = []
    for num in list_of_trial_nums:
        if num < 10:
            mystrings.append('T0' + str(num))
        else:
            mystrings.append('T' + str(num))
    return mystrings


def convert_subject_nums_to_strings(list_of_subject_nums: int):
    mystrings = []
    for num in list_of_subject_nums:
        if num < 10:
            mystrings.append('S0' + str(num))
        else:
            mystrings.append('S' + str(num))
    return mystrings

def custom_loss(y_actual, y_pred):
    mask = kb.greater(y_actual, 0)
    mask = tf.cast(mask, tf.float32)
    custom_loss = tf.math.reduce_sum(
        kb.square(mask*(y_actual-y_pred)))/tf.math.reduce_sum(mask)
    return custom_loss

def load_file(myfile):
    gait_phase_to_get = 'TM_Stance_Phase'
    df = pd.read_csv(myfile, usecols=['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y',
                     'gyro_z', 'ankle_angle', 'ankle_velocity', 'TM_Is_Stance_Phase', gait_phase_to_get])
    
    first_heel_strike_index_found = False
    
    for i in range(len(df)):
        if df['TM_Is_Stance_Phase'][i] == 1 and not first_heel_strike_index_found:
            first_heel_strike_index, first_heel_strike_index_found = i, True
            break
    
    if (not first_heel_strike_index_found): raise CustomError('Something is Wrong with the Left & Right File!!!')
    
    df = df[first_heel_strike_index:df.index[-1]]
    # if 'LEFT' in myfile:
    #     df.insert(8, 'is_left', value=1)
    # else:
    #     df.insert(8, 'is_left', value=0)
    return df.values


def leave_one_sub_out(window_size, model, sub_left_out, do_save=True, num_epochs=10):
    # Experiment_options
    num_snapshots_in_sequence = 128
    num_channels = 8
    use_smooth = False
    subject_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    sequence_len = num_snapshots_in_sequence + window_size - 1
    training_instances = np.empty(
        shape=[0, sequence_len, 11], dtype=np.float32)
    valid_instances = np.empty(
        shape=[0, sequence_len, 11], dtype=np.float32)
    files_to_load = get_files_to_use(
        "subject_data_processed", subject_list,
        sides=['LEFT', 'RIGHT'], trial_nums=1+np.arange(12))

    print('Loading subject data...')
    for subject_num in subject_list:
        for myfile in get_files_to_use(
            "subject_data_processed", [subject_num],
                sides=['LEFT', 'RIGHT'], trial_nums=1+np.arange(12)):
            data = load_file(myfile, use_smooth=use_smooth)
            num_rows, num_cols = data.shape
            num_rows_to_drop = num_rows % sequence_len

            data = data[0:-num_rows_to_drop]
            new_num_rows, num_cols = data.shape

            num_sequences = new_num_rows/sequence_len
            new_data_shape = (int(num_sequences), sequence_len, num_cols)
            new_instances = data.reshape(new_data_shape)
            if subject_num == sub_left_out:
                valid_instances = np.append(
                    valid_instances, new_instances, axis=0)
            else:
                training_instances = np.append(
                    training_instances, new_instances, axis=0)

    shuffled_training_instances = tf.random.shuffle(training_instances)
    x_train = tf.cast(
        shuffled_training_instances[:, :, :num_channels], tf.float32)
    y_gp_train = shuffled_training_instances[:, window_size-1:, -1]
    y_ss_train = shuffled_training_instances[:, window_size-1:, -2]

    x_valid = tf.cast(valid_instances[:, :, :num_channels], tf.float32)
    y_gp_valid = valid_instances[:, window_size-1:, -1]
    y_ss_valid = valid_instances[:, window_size-1:, -2]

    # BUILD MODEL
    tf.keras.backend.clear_session()

    fname = 'default_model_S' + str(sub_left_out) + '_leftout'

    csv_logger = CSVLogger('saved_models/' + fname +
                           '_log.csv', separator=',', append=False)
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True)
    mc = tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    history = model.fit(x=x_train, y=[y_gp_train, y_ss_train],
                        batch_size=32, epochs=num_epochs, validation_data=(x_valid, [y_gp_valid, y_ss_valid]),
                        callbacks=[es, csv_logger], verbose=1)
    if do_save:
        model.save('saved_models/' + fname + '.h5')


def train_on_all(model, window_size, num_epochs=12, do_save=True, use_smooth=False):
    # Experiment_options
    num_snapshots_in_sequence = 128
    num_channels = 8
    subject_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    sequence_len = num_snapshots_in_sequence + window_size - 1
    training_instances = np.empty(
        shape=[0, sequence_len, 11], dtype=np.float32)
    files_to_load = get_files_to_use(
        "subject_data_processed", subject_list,
        sides=['LEFT', 'RIGHT'], trial_nums=1+np.arange(12))

    print('Loading subject data...')
    for subject_num in subject_list:
        print('loading sub: ', subject_num)
        for myfile in get_files_to_use(
            "subject_data_processed", [subject_num],
                sides=['LEFT', 'RIGHT'], trial_nums=1+np.arange(12)):
            data = load_file(myfile, use_smooth=use_smooth)
            num_rows, num_cols = data.shape
            num_rows_to_drop = num_rows % sequence_len

            data = data[0:-num_rows_to_drop]
            new_num_rows, num_cols = data.shape

            num_sequences = new_num_rows/sequence_len
            new_data_shape = (int(num_sequences), sequence_len, num_cols)
            new_instances = data.reshape(new_data_shape)
            training_instances = np.append(
                training_instances, new_instances, axis=0)

    shuffled_training_instances = tf.random.shuffle(training_instances)
    x_train = tf.cast(
        shuffled_training_instances[:, :, :num_channels], tf.float32)
    y_gp_train = shuffled_training_instances[:, window_size-1:, -1]
    y_ss_train = shuffled_training_instances[:, window_size-1:, -2]

    # BUILD MODEL
    tf.keras.backend.clear_session()

    fname = 'model_trained_on_all_subs'

    csv_logger = CSVLogger('saved_models/' + fname +
                           '_log.csv', separator=',', append=False)
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True)
    mc = tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    history = model.fit(x=x_train, y=[y_gp_train, y_ss_train],
                        batch_size=32, epochs=num_epochs,
                        callbacks=[es, csv_logger], verbose=1)
    if do_save:
        model.save('saved_models/' + fname + '.h5')


def evaluate_model(model, window_size, subject_to_test, use_smooth):
    '''Test model with whole trial at once.'''
    use_side_flag = False
    num_channels = 8

    trials_to_test = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    sides_to_test = ['LEFT']
    files_to_test = get_files_to_use('subject_data_processed', subject_nums=[
        subject_to_test], sides=sides_to_test, trial_nums=trials_to_test)
    # print(files_to_test)

    rmses = []
    gp_results = []
    ss_results = []
    all_stance_err = np.empty(1)
    all_stance_gp_true = np.empty(1)
    for myfile in files_to_test:
        data = load_file(myfile, use_smooth=use_smooth)
        x_test = tf.expand_dims(data[:, :num_channels], axis=0)
        x_test = tf.cast(x_test, dtype=tf.float32)
        y_gp_test = data[window_size-1:, -1]
        y_ss_test = data[window_size-1:, -2]

        y_gp_test_yo = tf.expand_dims(y_gp_test, axis=0)
        y_ss_test_yo = tf.expand_dims(y_ss_test, axis=0)
        results = model.evaluate(x=x_test, y=(
            y_gp_test_yo, y_ss_test_yo), verbose=0)
        gp_results.append(np.sqrt(results[1]))
        ss_results.append(results[2])

    # print('RSME_gp:', np.mean(gp_results), np.mean(
    #     rmses), len(gp_results), len(rmses))
    # print('mean error_ss:', np.mean(ss_results))
    return gp_results, ss_results


if __name__ == '__main__':
    history = leave_one_sub_out()
    plt.plot(history.history['val_gait_phase_output_loss'])
    plt.show()
