# omniglot_loader
import os
import random
import numpy as np
import math
from PIL import Image

from image_augmentor import ImageAugmentor


class OmniglotLoader:
    def __init__(self, dataset_path, use_augmentation, batch_size):
        self.dataset_path = dataset_path
        # self.train_dictionary = {}
        self.evaluation_dictionary = {}
        self.image_width = 105
        self.image_height = 105
        self.batch_size = batch_size
        self.use_augmentation = use_augmentation
        self._train_alphabets = []
        self._validation_images = []
        self._evaluation_images = []
        self._current_train_class_index = 0
        self._current_validation_label_index = 0
        self._current_evaluation_label_index = 0

        self.load_dataset()

        if (self.use_augmentation):
            self.image_augmentor = self.createAugmentor()
        else:
            self.use_augmentation = []

    # def load_alphabet_dictionary(self, alphabet_path):
    #     alphabet_dictionary = {}
    #     for character in os.listdir(alphabet_path):
    #         character_path = os.path.join(alphabet_path, character)
    #         alphabet_dictionary[character] = os.listdir(character_path)
    #     return alphabet_dictionary


    def load_dataset(self):
        # train_path = '/content/Siamese-Networks-for-One-Shot-Learning/Omniglot Dataset/images_background'
        # validation_path = '/content/Siamese-Networks-for-One-Shot-Learning/Omniglot Dataset/images_evaluation'
        train_path = os.path.join(self.dataset_path, 'train')
        validation_path = os.path.join(self.dataset_path, 'valid')
        # self.train_dictionary = {alphabet: self.load_alphabet_dictionary(os.path.join(train_path, alphabet)) for alphabet in os.listdir(train_path)}
        # self.evaluation_dictionary = {alphabet: self.load_alphabet_dictionary(os.path.join(validation_path, alphabet)) for alphabet in os.listdir(validation_path)}

    def get_random_image_path(self, current_class, image_indexes):
        image_folder_name = 'train'
        if self.use_augmentation:
            image_folder_name += '_augmented'

        image_path = os.path.join(self.dataset_path, 'split_data_copy', 'train', current_class)
        # return os.path.join(image_path, self.train_dictionary[current_alphabet][current_character][image_indexes[0]])
        return os.path.join(image_path, self.train_dictionary[image_indexes[0]])

    def get_support_set_images(self, current_alphabet, different_characters, number_of_support_characters):
        support_images = []

        for _ in range(number_of_support_characters - 1):
            current_character = random.choice(different_characters)
            available_images = self.train_dictionary[current_alphabet][current_character]
            image_indexes = random.sample(range(0, 20), 1)
            image_path = self.get_random_image_path(current_alphabet, current_character, image_indexes)
            support_images.append(image_path)

        return support_images

    def get_one_shot_batch(self, support_set_size, is_validation):
        if is_validation:
            alphabets = self._validation_images
            current_alphabet_index = self._current_validation_label_index
            image_folder_name = 'train'
            dictionary = self.train_dictionary
        else:
            alphabets = self._evaluation_images
            current_alphabet_index = self._current_evaluation_label_index
            image_folder_name = 'train'
            dictionary = self.evaluation_dictionary

        current_alphabet = alphabets[current_alphabet_index]
        available_characters = list(dictionary[current_alphabet].keys())
        number_of_characters = len(available_characters)

        bacth_images_path = []

        test_character_index = random.sample(
            range(0, number_of_characters), 1)

        # Get test image
        current_character = available_characters[test_character_index[0]]

        available_images = (dictionary[current_alphabet])[current_character]

        image_indexes = random.sample(range(0, 20), 2)
        image_path = os.path.join(
            self.dataset_path, image_folder_name, current_alphabet, current_character)

        test_image = os.path.join(
            image_path, available_images[image_indexes[0]])
        bacth_images_path.append(test_image)
        image = os.path.join(
            image_path, available_images[image_indexes[1]])
        bacth_images_path.append(image)

        # Let's get our test image and a pair corresponding to
        if support_set_size == -1:
            number_of_support_characters = number_of_characters
        else:
            number_of_support_characters = support_set_size

        different_characters = available_characters[:]
        different_characters.pop(test_character_index[0])

        # There may be some alphabets with less than 20 characters
        if number_of_characters < number_of_support_characters:
            number_of_support_characters = number_of_characters

        support_characters_indexes = random.sample(
            range(0, number_of_characters - 1), number_of_support_characters - 1)

        for index in support_characters_indexes:
            current_character = different_characters[index]
            available_images = (dictionary[current_alphabet])[
                current_character]
            image_path = os.path.join(
                self.dataset_path, image_folder_name, current_alphabet, current_character)

            image_indexes = random.sample(range(0, 20), 1)
            image = os.path.join(
                image_path, available_images[image_indexes[0]])
            bacth_images_path.append(test_image)
            bacth_images_path.append(image)

        images, labels = self._convert_path_list_to_images_and_labels(
            bacth_images_path, is_one_shot_task=True)

        return images, labels

    def createAugmentor(self):
        rotation_range = [-15, 15]
        shear_range = [-0.3 * 180 / math.pi, 0.3 * 180 / math.pi]
        zoom_range = [0.8, 2]
        shift_range = [5, 5]

        return ImageAugmentor(0.5, shear_range, rotation_range, shift_range, zoom_range)

    def split_train_datasets(self):

        available_alphabets = list(self.train_dictionary)
        number_of_alphabets = len(available_alphabets)

        train_indexes = random.sample(
            range(0, number_of_alphabets - 1), int(0.8 * number_of_alphabets))

        # If we sort the indexes in reverse order we can pop them from the list
        # and don't care because the indexes do not change
        train_indexes.sort(reverse=True)

        for index in train_indexes:
            self._train_alphabets.append(available_alphabets)
            available_alphabets.pop(index)

        # The remaining alphabets are saved for validation
        self._validation_images = available_alphabets
        self._evaluation_images= list(self.evaluation_dictionary.keys())

    def _convert_path_list_to_images_and_labels(self, path_list, is_one_shot_task):
        number_of_pairs = int(len(path_list) / 2)
        pairs_of_images = [np.zeros(
            (number_of_pairs, self.image_height, self.image_height, 1)) for i in range(2)]
        labels = np.zeros((number_of_pairs, 1))

        for pair in range(number_of_pairs):
            try:
                image = Image.open(path_list[pair * 2]).convert('L')
                image = image.resize((105,105))
                image = np.asarray(image).astype(np.float64)
                image = image / image.std() - image.mean()
                pairs_of_images[0][pair, :, :, 0] = image

                image = Image.open(path_list[pair * 2 + 1]).convert('L')
                image = image.resize((105, 105))
                image = np.asarray(image).astype(np.float64)
                image = image / image.std() - image.mean()
                image_resized = np.zeros((105, 105))
                image_resized[:image.shape[0], :image.shape[1]] = image
                pairs_of_images[1][pair, :, :, 0] = image

                if not is_one_shot_task:
                    if (pair + 1) % 2 == 0:
                        labels[pair] = 0
                    else:
                        labels[pair] = 1
                else:
                    if pair == 0:
                        labels[pair] = 1
                    else:
                        labels[pair] = 0
            except FileNotFoundError as e:
                print(f"Error processing image pair {pair}: {e}")


        if not is_one_shot_task:
            try:
                random_permutation = np.random.permutation(number_of_pairs)
                labels = labels[random_permutation]
                pairs_of_images[0][:, :, :, :] = pairs_of_images[0][random_permutation, :, :, :]
                pairs_of_images[1][:, :, :, :] = pairs_of_images[1][random_permutation, :, :, :]
            except Exception as e:
                print(f"Error during random permutation: {e}")

        return pairs_of_images, labels

    def get_train_batch(self):
        # current_alphabet = self._train_alphabets[self._current_train_alphabet_index]
        # available_characters = list(
        #     self.train_dictionary[current_alphabet].keys())
        # available_characters = list(
        #          self.train_dictionary[current_alphabet].keys())
        # number_of_class = len(available_characters)
        # 현재 알파벳에 대해 사용 가능한 클래스(character)들을 확인
        current_train_path = os.path.join(self.dataset_path, 'train')
        available_classes = os.listdir(current_train_path)

        # 클래스(character)들에 대한 딕셔너리 생성
        # class_dictionary = {class_name: os.path.join(current_train_path, class_name, 'images') for class_name in
        #                     available_classes}
        class_dictionary = {class_name: [os.path.join(current_train_path, class_name, image_name)
                                         for image_name in
                                         os.listdir(os.path.join(current_train_path, class_name))]
                            for class_name in available_classes}
        # class_dictionary = {class_name: [os.path.join(current_train_path, class_name, image_name)
        #                                  for image_name in
        #                                  os.listdir(os.path.join(current_train_path, class_name))
        #                                  if not image_name.startswith('.')]  # 숨김 파일 건너뛰기
        #                     for class_name in os.listdir(current_train_path)
        #                     if not class_name.startswith('.')}  # 클래스 디렉토리의 숨김 파일도 건너뛰기

        # 클래스(character)들의 수
        number_of_class = len(available_classes)

        batch_images_path = []

        # If the number of classes is less than self.batch_size/2
        # we have to repeat characters
        selected_characters_indexes = [random.randint(
            0, number_of_class - 1) for i in range(self.batch_size)]

        # for index in selected_characters_indexes:
        #     current_class = available_classes[index]
        #     class_images_path = class_dictionary[current_class]
        #
        #     # Randomly select 3 indexes of images from the same class (Remember
        #     # that for each class we have 20 examples).
        #     image_indexes = random.sample(range(0, 20), 3)
        #
        #     for i in range(2):
        #         image = os.path.join(
        #             class_images_path, str(image_indexes[i]) + '.jpg')
        #         batch_images_path.append(image)
        #
        #     # Now let's take care of the pair of images from different classes
        #     image = os.path.join(
        #         class_images_path, str(image_indexes[2]) + '.jpg')
        #     batch_images_path.append(image)
        #
        #     different_classes = available_classes[:]
        #     different_classes.pop(index)
        #     different_class_index = random.sample(
        #         range(0, number_of_class - 1), 1)
        #     different_class = different_classes[different_class_index[0]]
        #     different_class_images_path = class_dictionary[different_class]
        #
        #     image_indexes = random.sample(range(0, 20), 1)
        #     image = os.path.join(
        #         different_class_images_path, str(image_indexes[0]) + '.png')
        #     batch_images_path.append(image)

        for index in selected_characters_indexes:
            current_class = available_classes[index]
            class_images_path = class_dictionary[current_class]

            # Randomly select 3 indexes of images from the same class (Remember
            # that for each class we have 20 examples).
            image_indexes = random.sample(range(0, 20), 3)

            for i in range(2):
                # 수정된 부분: 이미지 경로를 직접 리스트에서 가져옵니다.
                image = class_images_path[image_indexes[i]]
                batch_images_path.append(image)

            # Now let's take care of the pair of images from different classes
            # 수정된 부분: 이미지 경로를 직접 리스트에서 가져옵니다.
            image = class_images_path[image_indexes[2]]
            batch_images_path.append(image)

            different_classes = available_classes[:]
            different_classes.pop(index)
            different_class_index = random.sample(
                range(0, number_of_class - 1), 1)
            different_class = different_classes[different_class_index[0]]
            different_class_images_path = class_dictionary[different_class]

            image_indexes = random.sample(range(0, 20), 1)
            # 수정된 부분: 이미지 경로를 직접 리스트에서 가져옵니다.
            image = different_class_images_path[image_indexes[0]]
            batch_images_path.append(image)

        self._current_train_class_index += 1

        if (self._current_train_class_index > 23):
            self._current_train_class_index = 0

        images, labels = self._convert_path_list_to_images_and_labels(
            batch_images_path, is_one_shot_task=False)

        # Get random transforms if augmentation is on
        if self.use_augmentation:
            images = self.image_augmentor.get_random_transform(images)

        return images, labels


    def get_one_shot_batch(self, support_set_size, is_validation):
        if is_validation:
            alphabets = self._validation_alphabets
            current_alphabet_index = self._current_validation_alphabet_index
            image_folder_name = 'train'
            dictionary = self.train_dictionary
        else:
            alphabets = self._evaluation_alphabets
            current_alphabet_index = self._current_evaluation_alphabet_index
            image_folder_name = 'valid'
            dictionary = self.evaluation_dictionary

        current_alphabet = alphabets[current_alphabet_index]
        available_characters = list(dictionary[current_alphabet].keys())
        number_of_characters = len(available_characters)

        bacth_images_path = []

        test_character_index = random.sample(
            range(0, number_of_characters), 1)

        # Get test image
        current_character = available_characters[test_character_index[0]]

        available_images = (dictionary[current_alphabet])[current_character]

        image_indexes = random.sample(range(0, 20), 2)
        image_path = os.path.join(
            self.dataset_path, image_folder_name, current_alphabet, current_character)

        test_image = os.path.join(
            image_path, available_images[image_indexes[0]])
        bacth_images_path.append(test_image)
        image = os.path.join(
            image_path, available_images[image_indexes[1]])
        bacth_images_path.append(image)

        # Let's get our test image and a pair corresponding to
        if support_set_size == -1:
            number_of_support_characters = number_of_characters
        else:
            number_of_support_characters = support_set_size

        different_characters = available_characters[:]
        different_characters.pop(test_character_index[0])

        # There may be some alphabets with less than 20 characters
        if number_of_characters < number_of_support_characters:
            number_of_support_characters = number_of_characters

        support_characters_indexes = random.sample(
            range(0, number_of_characters - 1), number_of_support_characters - 1)

        for index in support_characters_indexes:
            current_character = different_characters[index]
            available_images = (dictionary[current_alphabet])[
                current_character]
            image_path = os.path.join(
                self.dataset_path, image_folder_name, current_alphabet, current_character)

            image_indexes = random.sample(range(0, 20), 1)
            image = os.path.join(
                image_path, available_images[image_indexes[0]])
            bacth_images_path.append(test_image)
            bacth_images_path.append(image)

        images, labels = self._convert_path_list_to_images_and_labels(
            bacth_images_path, is_one_shot_task=True)

        return images, labels

    def one_shot_test(self, model, support_set_size, number_of_tasks_per_alphabet,
                      is_validation):
        # Set some variables that depend on dataset
        if is_validation:
            labels = self._validation_images
            print('\nMaking One Shot Task on validation images:')
        else:
            labels = self._evaluation_images
            print('\nMaking One Shot Task on evaluation images:')

        mean_global_accuracy = 0

        for label in labels:
            mean_label_accuracy = 0
            for _ in range(number_of_tasks_per_alphabet):
                images, _ = self.get_one_shot_batch(
                    support_set_size, is_validation=is_validation)
                probabilities = model.predict_on_batch(images)

                # Added this condition because noticed that sometimes the outputs
                # of the classifier was almost the same in all images, meaning that
                # the argmax would be always by defenition 0.
                if np.argmax(probabilities) == 0 and probabilities.std()>0.01:
                    accuracy = 1.0
                else:
                    accuracy = 0.0

                mean_label_accuracy += accuracy
                mean_global_accuracy += accuracy

            mean_label_accuracy /= number_of_tasks_per_alphabet

            print(label + ' class' + ', accuracy: ' +
                  str(mean_label_accuracy))
            if is_validation:
                self._current_validation_label_index += 1
            else:
                self._current_evaluation_label_index += 1

        mean_global_accuracy /= max(1, len(labels) * number_of_tasks_per_alphabet)


        print('\nMean global accuracy: ' + str(mean_global_accuracy))

        # reset counter
        if is_validation:
            self._current_validation_label_index = 0
        else:
            self._current_evaluation_label_index = 0

        return mean_global_accuracy
