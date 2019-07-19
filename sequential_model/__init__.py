from sequential_model.model import SeqModel
import matplotlib.pyplot as plt

if __name__ == '__main__':
    seq = SeqModel()
    #(x, y) = seq.make_random_data()
    #plt.plot(x, y, 'o')
    #plt.show()
    seq.create_model()
    seq.execute()
    seq.load_model()