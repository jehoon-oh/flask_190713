from mnist_test.mnist_load import MnistTest
import matplotlib.pyplot as plt

if __name__ == '__main__':
    mtest = MnistTest()
    model = mtest.create_model()
    # predictions_array, true_label, img
    predictions_array = model[0]
    true_label = model[1]
    img = model[2]

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    test_idx = 0
    print(class_names[true_label[test_idx]])
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    mtest.plot_image(test_idx, predictions_array, true_label, img)
    plt.subplot(1, 2, 2)
    mtest.plot_value_array(test_idx, predictions_array, true_label)
    plt.show()



    test_idx = 3
    print(class_names[true_label[test_idx]])
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    mtest.plot_image(test_idx, predictions_array, true_label, img)
    plt.subplot(1, 2, 2)
    mtest.plot_value_array(test_idx, predictions_array, true_label)
    plt.show()

    test_idx = 12
    print(class_names[true_label[test_idx]])
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    mtest.plot_image(test_idx, predictions_array, true_label, img)
    plt.subplot(1, 2, 2)
    mtest.plot_value_array(test_idx, predictions_array, true_label)
    plt.show()

    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        mtest.plot_image(i, predictions_array, true_label, img)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        mtest.plot_value_array(i, predictions_array, true_label)
    plt.show()