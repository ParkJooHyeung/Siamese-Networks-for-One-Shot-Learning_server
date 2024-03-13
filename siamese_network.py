# Siamese_network with Graph
import os

from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, Subtract, Lambda, BatchNormalization, Dropout
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from tensorflow.keras import backend as K

import tensorflow as tf

import matplotlib.pyplot as plt

from omniglot_loader import OmniglotLoader
from modified_sgd import Modified_SGD

class SiameseNetwork:
    """Class that constructs the Siamese Net for training

    This Class was constructed to create the siamese net and train it.

    Attributes:
        input_shape: image size
        model: current siamese model
        learning_rate: SGD learning rate
        omniglot_loader: instance of OmniglotLoader
        summary_writer: tensorflow writer to store the logs
    """

    def __init__(self, dataset_path,  learning_rate, batch_size, use_augmentation,
                 learning_rate_multipliers, l2_regularization_penalization, tensorboard_log_path):
        """Inits SiameseNetwork with the provided values for the attributes.

        It also constructs the siamese network architecture, creates a dataset
        loader and opens the log file.

        Arguments:
            dataset_path: path of Omniglot dataset
            learning_rate: SGD learning rate
            batch_size: size of the batch to be used in training
            use_augmentation: boolean that allows us to select if data augmentation
                is used or not
            learning_rate_multipliers: learning-rate multipliers (relative to the learning_rate
                chosen) that will be applied to each fo the conv and dense layers
                for example:
                    # Setting the Learning rate multipliers
                    LR_mult_dict = {}
                    LR_mult_dict['conv1']=1
                    LR_mult_dict['conv2']=1
                    LR_mult_dict['dense1']=2
                    LR_mult_dict['dense2']=2
            l2_regularization_penalization: l2 penalization for each layer.
                for example:
                    # Setting the Learning rate multipliers
                    L2_dictionary = {}
                    L2_dictionary['conv1']=0.1
                    L2_dictionary['conv2']=0.001
                    L2_dictionary['dense1']=0.001
                    L2_dictionary['dense2']=0.01
            tensorboard_log_path: path to store the logs
        """
        self.input_shape = (105, 105, 1)  # Size of images
        self.model = []
        self.learning_rate = learning_rate
        self.omniglot_loader = OmniglotLoader(
            dataset_path=dataset_path, use_augmentation=use_augmentation, batch_size=batch_size)
        # self.summary_writer = tf.summary.FileWriter(tensorboard_log_path)
        self._construct_siamese_architecture(learning_rate_multipliers,
                                              l2_regularization_penalization)
        log_dir = "/content/drive/MyDrive/logs"  # 로그 디렉토리는 적절하게 변경 가능
        self.summary_writer = tf.summary.create_file_writer(log_dir)
        self.current_iteration = 0  # 현재 iteration을 저장할 변수 추가

        self.current_iteration = 0  # 현재 iteration을 저장할 변수 추가
        self.train_losses = []  # Train losses for each iteration
        self.train_accuracies = []  # Train accuracies for each iteration

    def _construct_siamese_architecture(self, learning_rate_multipliers,
                                         l2_regularization_penalization):
        """ Constructs the siamese architecture and stores it in the class

        Arguments:
            learning_rate_multipliers
            l2_regularization_penalization
        """

        # Let's define the cnn architecture
        convolutional_net = Sequential()
        convolutional_net.add(Conv2D(filters=64, kernel_size=(10, 10),
                                     activation='relu',
                                     input_shape=self.input_shape,
                                     kernel_regularizer=l2(
                                         l2_regularization_penalization['Conv1']),
                                     name='Conv1'))
        convolutional_net.add(MaxPool2D())

        convolutional_net.add(Conv2D(filters=128, kernel_size=(7, 7),
                                     activation='relu',
                                     kernel_regularizer=l2(
                                         l2_regularization_penalization['Conv2']),
                                     name='Conv2'))
        convolutional_net.add(MaxPool2D())

        convolutional_net.add(Conv2D(filters=128, kernel_size=(4, 4),
                                     activation='relu',
                                     kernel_regularizer=l2(
                                         l2_regularization_penalization['Conv3']),
                                     name='Conv3'))
        convolutional_net.add(MaxPool2D())

        convolutional_net.add(Conv2D(filters=256, kernel_size=(4, 4),
                                     activation='relu',
                                     kernel_regularizer=l2(
                                         l2_regularization_penalization['Conv4']),
                                     name='Conv4'))

        convolutional_net.add(Flatten())
        convolutional_net.add(
            Dense(units=4096, activation='sigmoid',
                  kernel_regularizer=l2(
                      l2_regularization_penalization['Dense1']),
                  name='Dense1'))
        convolutional_net.add(BatchNormalization())
        # Dropout 추가
        convolutional_net.add(Dropout(0.5))

        # Now the pairs of images
        input_image_1 = Input(self.input_shape)
        input_image_2 = Input(self.input_shape)

        encoded_image_1 = convolutional_net(input_image_1)
        encoded_image_2 = convolutional_net(input_image_2)

        # L1 distance layer between the two encoded outputs
        # One could use Subtract from Keras, but we want the absolute value
        l1_distance_layer = Lambda(
            lambda tensors: K.abs(tensors[0] - tensors[1]))
        l1_distance = l1_distance_layer([encoded_image_1, encoded_image_2])

        # Same class or not prediction
        prediction = Dense(units=1, activation='sigmoid')(l1_distance)
        self.model = Model(
            inputs=[input_image_1, input_image_2], outputs=prediction)

        # Define the optimizer and compile the model
        optimizer = Modified_SGD(
            learning_rate=self.learning_rate,
            lr_multipliers=learning_rate_multipliers,
            momentum=0.5)
        # optimizer = 'adam'

        self.model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'],
                           optimizer=optimizer)

        # 훈련 손실 및 정확도를 저장할 리스트 초기화
        self.train_losses = []
        self.train_accuracies = []

    def _write_logs_to_tensorboard(self, current_iteration, validation_accuracy, evaluate_each, train_loss, train_accuracy):
        """텐서보드 로그를 작성하고 그래프를 그립니다."""
        # 손실 및 정확도를 리스트에 추가합니다
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_accuracy)

        with self.summary_writer.as_default():
            # 손실과 정확도를 텐서보드에 기록
            tf.summary.scalar('Train Loss', train_loss, step=current_iteration)
            tf.summary.scalar('Train Accuracy', train_accuracy, step=current_iteration)
            tf.summary.scalar('One-Shot Validation Accuracy', validation_accuracy, step=current_iteration)

            # summary_writer로 내용을 즉시 쓰기
            self.summary_writer.flush()

        # 그래프 그리기
        self._plot_graphs(current_iteration, self.train_losses, self.train_accuracies, validation_accuracy, evaluate_each)


    def _plot_graphs(self, current_iteration, train_losses, train_accuracies, validation_accuracy, evaluate_each):
        """ 손실 및 정확도에 대한 그래프를 그립니다.

        Arguments:
            current_iteration: x축에 표시될 반복 횟수
            train_losses: 훈련 손실의 리스트
            train_accuracies: 훈련 정확도의 리스트
            validation_accuracy: 현재 원샷 검증 정확도
            evaluate_each: evaluate_each 반복마다 기록된 리스트
        """
        iterations = list(range(current_iteration - len(train_losses) + 1, current_iteration + 1))

        # 훈련 손실 플로팅
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 2)
        plt.plot(iterations, train_losses, label='Train Loss', marker='o', color='blue')
        plt.title('Train Loss over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()

        # 훈련 정확도 플로팅
        plt.subplot(1, 2, 2)
        plt.plot(iterations, train_accuracies, label='Train Accuracy', marker='o', color='red')
        plt.title('Train Accuracy over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.legend()

        # 플롯 보여주기
        plt.tight_layout()
        plt.show()

    def train_siamese_network(self, number_of_iterations, support_set_size,
                              final_momentum, momentum_slope, evaluate_each,
                              model_name):
        # 먼저 30개의 학습 알파벳을 24개는 학습, 6개는 검증으로 무작위로 나눕니다.
        # self.omniglot_loader.split_train_datasets()

        # evaluate_each 번 반복마다 tensorboard 로그에 전달될 100번의 반복 손실 및 정확도를 저장할 변수
        self.train_losses = []
        self.train_accuracies = []
        count = 0
        early_stop = 0  # Fix typo: earrly_stop -> early_stop
        # 조기 중지 기준 변수
        best_validation_accuracy = 0.0
        best_accuracy_iteration = 0
        validation_accuracy = 0.0

        for iteration in range(number_of_iterations):
            images, labels = self.omniglot_loader.get_train_batch()
            try:
                train_loss, train_accuracy = self.model.train_on_batch(images, labels)
            except Exception as e:
                print(f"Exception during training on batch: {e}")
                continue
            if (iteration + 1) % 500 == 0:
                K.set_value(self.model.optimizer.lr, K.get_value(
                    self.model.optimizer.lr) * 0.99)
            if K.get_value(self.model.optimizer.momentum) < final_momentum:
                new_momentum_value = K.get_value(self.model.optimizer.momentum) + momentum_slope
                self.model.optimizer.momentum = new_momentum_value

            # validation set
            count += 1
            print('Iteration %d/%d: Train loss: %f, Train Accuracy: %f, lr = %f' %
                  (iteration + 1, number_of_iterations, train_loss, train_accuracy, K.get_value(self.model.optimizer.lr)))

            # 각 100번 반복마다 one_shot_task 수행 및 저장된 손실 및 정확도를 tensorboard에 기록
            if (iteration + 1) % evaluate_each == 0:
                number_of_runs_per_alphabet = 40
                validation_accuracy = self.omniglot_loader.one_shot_test(
                    self.model, support_set_size, number_of_runs_per_alphabet, is_validation=True)

                self._write_logs_to_tensorboard(iteration + 1, validation_accuracy, evaluate_each, train_loss, train_accuracy)
                count = 0

                # 일부 초매개변수는 100%로 이끄는데, 출력은 거의 모든 이미지에서 동일합니다.
                if (validation_accuracy == 1.0 and train_accuracy == 0.5):
                    print('Early Stopping: Gradient Explosion')
                    print('Validation Accuracy = ' + str(best_validation_accuracy))
                    return 0
                elif train_accuracy == 0.0:
                    return 0
                else:
                    # 모델 저장
                    if validation_accuracy > best_validation_accuracy:
                        best_validation_accuracy = validation_accuracy
                        best_accuracy_iteration = iteration

                        model_json = self.model.to_json()

                        if not os.path.exists('./models'):
                            os.makedirs('./models')
                        with open('models/' + model_name + '.json', "w") as json_file:
                            json_file.write(model_json)
                        self.model.save_weights('models/' + model_name + '.h5')

            # 1만 번 동안 정확도가 향상되지 않으면 훈련 중지
            if iteration - best_accuracy_iteration > 10000:
                print('Early Stopping: validation accuracy did not increase for 10000 iterations')
                print('Best Validation Accuracy = ' + str(best_validation_accuracy))
                print('Validation Accuracy = ' + str(best_validation_accuracy))
                break

        self.train_losses = np.array([])
        self.train_accuracies = np.array([])

        print('Trained Ended!')
        return best_validation_accuracy
