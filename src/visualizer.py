import logging, numpy as np, seaborn as sns, matplotlib.pyplot as plt
import os

from . import utils as U


class Visualizer():
    def __init__(self, args):
        self.args = args
        U.set_logging(args)
        logging.info('')
        logging.info('Starting visualizing ...')

        self.action_names = {}
        self.action_names['ntu'] = [
            'drink water 1', 'eat meal/snack 2', 'brushing teeth 3', 'brushing hair 4', 'drop 5', 'pickup 6',
            'throw 7', 'sitting down 8', 'standing up 9', 'clapping 10', 'reading 11', 'writing 12',
            'tear up paper 13', 'wear jacket 14', 'take off jacket 15', 'wear a shoe 16', 'take off a shoe 17',
            'wear on glasses 18','take off glasses 19', 'put on a hat/cap 20', 'take off a hat/cap 21', 'cheer up 22',
            'hand waving 23', 'kicking something 24', 'put/take out sth 25', 'hopping 26', 'jump up 27',
            'make a phone call 28', 'playing with a phone 29', 'typing on a keyboard 30',
            'pointing to sth with finger 31', 'taking a selfie 32', 'check time (from watch) 33',
            'rub two hands together 34', 'nod head/bow 35', 'shake head 36', 'wipe face 37', 'salute 38',
            'put the palms together 39', 'cross hands in front 40', 'sneeze/cough 41', 'staggering 42', 'falling 43',
            'touch head 44', 'touch chest 45', 'touch back 46', 'touch neck 47', 'nausea or vomiting condition 48',
            'use a fan 49', 'punching 50', 'kicking other person 51', 'pushing other person 52',
            'pat on back of other person 53', 'point finger at the other person 54', 'hugging other person 55',
            'giving sth to other person 56', 'touch other person pocket 57', 'handshaking 58',
            'walking towards each other 59', 'walking apart from each other 60'
        ]
        self.action_names['cmu'] = [
            'walking 1', 'running 2', 'directing_traffic 3', 'soccer 4',
            'basketball 5', 'washwindow 6', 'jumping 7', 'basketball_signal 8'
        ]

        vocabulary_path = "/workspace/vocabulary.txt"
        if not os.path.exists(vocabulary_path):
            raise FileNotFoundError(f"Vocabulary file not found at: {vocabulary_path}")
        with open(vocabulary_path, "r") as f:
            vocabulary = [line.strip() for line in f.readlines()]
        self.action_names['mediapipe61'] = vocabulary

        self.font_sizes = {
            'ntu': 6,
            'cmu': 14,
            'mediapipe61': 10
        }


    def start(self):
        self.read_data()
        logging.info('Please select visualization function from follows: ')
        logging.info('1) wrong sample (ws), 2) important joints (ij), 3) NTU skeleton (ns),')
        logging.info('4) confusion matrix (cm), 5) action accuracy (ac)')
        while True:
            logging.info('Please input the number (or name) of function, q for quit: ')
            cmd = input(U.get_current_timestamp())
            if cmd in ['q', 'quit', 'exit', 'stop']:
                break
            elif cmd == '1' or cmd == 'ws' or cmd == 'wrong sample':
                self.show_wrong_sample()
            elif cmd == '2' or cmd == 'ij' or cmd == 'important joints':
                self.show_important_joints()
            elif cmd == '3' or cmd == 'ns' or cmd == 'NTU skeleton':
                self.show_NTU_skeleton()
            elif cmd == '4' or cmd == 'cm' or cmd == 'confusion matrix':
                self.show_confusion_matrix()
            elif cmd == '5' or cmd == 'ac' or cmd == 'action accuracy':
                self.show_action_accuracy()
            else:
                logging.info('Can not find this function!')
                logging.info('')


    def read_data(self):
        logging.info('Reading data ...')
        logging.info('')
        data_file = './visualization/extraction_{}.npz'.format(self.args.config)
        try:
            data = np.load(data_file, allow_pickle=True)
        except:
            data = None
            logging.info('')
            logging.error('Error: Wrong in loading this extraction file: {}!'.format(data_file))
            logging.info('Please extract the data first!')
            raise ValueError()
        logging.info('*********************Video Name************************')
        logging.info(data['name'][self.args.visualization_sample])
        logging.info('')

        feature = data['feature'][self.args.visualization_sample]
        self.location = data['location']
        if len(self.location) > 0:
            self.location = self.location[self.args.visualization_sample]
        self.data = data['data'][self.args.visualization_sample]
        self.label = data['label']
        weight = data['weight']
        out = data['out']
        cm = data['cm']
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        self.cm = np.divide(cm.astype('float'), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums!=0)

        dataset = self.args.dataset.split('-')[0]
        self.names = self.action_names[dataset]
        self.font_size = self.font_sizes[dataset]

        self.pred = np.argmax(out, 1)
        self.pred_class = self.pred[self.args.visualization_sample] + 1
        self.actural_class = self.label[self.args.visualization_sample] + 1
        if self.args.visualization_class == 0:
            self.args.visualization_class = self.actural_class
        self.probablity = out[self.args.visualization_sample, self.args.visualization_class-1]
        self.result = np.einsum('kc,ctvm->ktvm', weight, feature)
        self.result = self.result[self.args.visualization_class-1]


    def show_action_accuracy(self):
        cm = self.cm.round(4)
        logging.info('Accuracy of each class:')
        accuracy = cm.diagonal()
        for i in range(len(accuracy)):
            logging.info('{}: {}'.format(self.names[i], accuracy[i]))
        logging.info('')

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(self.names, accuracy, align='center')
        plt.xticks(fontsize=10, rotation=90)
        plt.yticks(fontsize=10)
        plt.title('Accuracy for Each Action')
        plt.ylabel('Accuracy')
        plt.tight_layout() # Adjust layout to prevent labels overlapping

        # 1. Save the plot
        output_path = f'action_accuracy_{self.args.config}.png'
        plt.savefig(output_path)
        logging.info(f'✅ Action accuracy plot saved to {output_path}')

        # 2. Show the plot
        plt.show()
        plt.close(fig) # Close the figure to free memory


    def show_confusion_matrix(self):
        cm = self.cm.round(2)
        show_name_x = range(1,len(self.names)+1)
        show_name_y = self.names
        font_size = self.font_size

        # Create plot
        fig, ax = plt.subplots(figsize=(18, 15))
        sns.heatmap(cm, cmap=plt.cm.Blues, annot=True, annot_kws={'fontsize':font_size-2}, cbar=False,
                    square=True, linewidths=0.1, linecolor='black', xticklabels=show_name_x, yticklabels=show_name_y, ax=ax)
        plt.xticks(fontsize=font_size, rotation=0)
        plt.yticks(fontsize=font_size)
        plt.xlabel('Predicted Class', fontsize=font_size+2)
        plt.ylabel('True Class', fontsize=font_size+2)
        plt.title('Confusion Matrix', fontsize=font_size+4)
        plt.tight_layout()

        # 1. Save the plot
        output_path = f'confusion_matrix_{self.args.config}.png'
        plt.savefig(output_path)
        logging.info(f'✅ Confusion matrix plot saved to {output_path}')

        # 2. Show the plot
        plt.show()
        plt.close(fig) # Close the figure to free memory


    def show_NTU_skeleton(self):
        if len(self.location) == 0:
            logging.info('This function is only for NTU dataset!')
            logging.info('')
            return

        C, T, V, M = self.location.shape
        connecting_joint = np.array([2,1,21,3,21,5,6,7,21,9,10,11,1,13,14,15,1,17,18,19,2,23,8,25,12]) - 1
        result = np.maximum(self.result, 0)
        result = result/np.max(result) if np.max(result) > 0 else result

        if len(self.args.visualization_frames) > 0:
            pause, frames = 0.5, self.args.visualization_frames
        else:
            pause, frames = 0.01, range(self.location.shape[1])

        fig = plt.figure(figsize=(8, 6))
        plt.ion()
        for t in frames:
            if np.sum(self.location[:,t,:,:]) == 0:
                continue

            plt.cla()
            plt.xlim(self.location[0].min(), self.location[0].max())
            plt.ylim(self.location[1].min(), self.location[1].max())
            plt.axis('off')
            plt.title('Frame: {} | Prob: {:.2f}% | Pred: {} | True: {}'.format(
                t, self.probablity*100, self.pred_class, self.actural_class
            ))

            for m in range(M):
                x = self.location[0,t,:,m]
                y = self.location[1,t,:,m]
                if np.sum(x) == 0 and np.sum(y) == 0:
                    continue

                c = []
                for v in range(V):
                    r = result[t//4,v,m] if t//4 < result.shape[0] else 0
                    g = 0
                    b = 1 - r
                    c.append([r,g,b])
                    k = connecting_joint[v]
                    plt.plot([x[v],x[k]], [y[v],y[k]], '-o', c=np.array([0.1,0.1,0.1]), linewidth=0.5, markersize=0)
                plt.scatter(x, y, marker='o', c=c, s=20)
            plt.pause(pause)
        plt.ioff()

        # 1. Save the final frame
        output_path = f'Skeleton_{self.args.config}_sample{self.args.visualization_sample}.png'
        plt.savefig(output_path, bbox_inches='tight')
        logging.info(f'✅ Skeleton plot saved to {output_path}')

        # 2. Show the final frame
        plt.show()
        plt.close(fig) # Close the figure to free memory


    def show_wrong_sample(self):
        wrong_sample = []
        for i in range(len(self.pred)):
            if not self.pred[i] == self.label[i]:
                wrong_sample.append(i)
        logging.info('*********************Wrong Sample Indices**********************')
        logging.info(wrong_sample)
        logging.info('')


    def show_important_joints(self):
        first_sum = np.sum(self.result[:,:,0], axis=0)
        first_index = np.argsort(-first_sum) + 1
        logging.info('*********************First Person**********************')
        logging.info('Weights of all joints:')
        logging.info(first_sum.round(4))
        logging.info('')
        logging.info('Most important joints (descending order):')
        logging.info(first_index)
        logging.info('')

        if self.result.shape[-1] > 1:
            second_sum = np.sum(self.result[:,:,1], axis=0)
            second_index = np.argsort(-second_sum) + 1
            logging.info('*********************Second Person*********************')
            logging.info('Weights of all joints:')
            logging.info(second_sum.round(4))
            logging.info('')
            logging.info('Most important joints (descending order):')
            logging.info(second_index)
            logging.info('')