import numpy as np
import math
import matplotlib.pyplot as plt
import pylab  # To extract the contours
import random
import sys, ast, getopt, types
import scipy.io

# --------------------------------------------------
# -- Data from the Berkeley dataset that contains --
# --  two labels that seems correct (visually..)  --
# --------------------------------------------------
mat_segmentation = {
    'val':{
        1: {'file': '130026.mat', 'segmentation': 1, 'description': "Croco"},
        2: {'file': '196073.mat', 'segmentation': 4, 'description': "Serpent"},
        3: {'file': '87046.mat', 'segmentation': 3, 'description': "Lezrad"}
    },
    'train':{
        1: {'file': '135037.mat', 'segmentation': 1, 'description': "Aigle"},
        2: {'file': '130034.mat', 'segmentation': 5, 'description': "Croco(herbe)"},
        3: {'file': '100098.mat', 'segmentation': 4, 'description': "Ours(mauvais)"}
    },
    'test':{
        1: {'file': '8068.mat', 'segmentation': 2, 'description': "Cygne(pasouf)"},
        2: {'file': '43051.mat', 'segmentation': 4, 'description': "Canard"},
        3: {'file': '3063.mat', 'segmentation': 3, 'description': "Avion(pasouf)"},
        4: {'file': '253092.mat', 'segmentation': 2, 'description': "Girafe"},
        5: {'file': '109055.mat', 'segmentation': 1, 'description': "Loup"},
        6: {'file': '108069.mat', 'segmentation': 1, 'description': "Tigre"},
        7: {'file': '69022.mat', 'segmentation': 4, 'description': "Kangourou"},
        8: {'file': '70011.mat', 'segmentation': 3, 'description': "Aigle(pasouf)"}
    }
}
# --------------------------------------------------
# --------------------------------------------------

def load_labels_from_Berkeley_dataset(name_file, segmentation_num):
    """ name_file = "13006.mat" for instance
        segmentation_num = number of the segmentation which contain only two labels
    """
    data = scipy.io.loadmat(name_file)
    return data['groundTruth'][0][segmentation_num - 1]['Segmentation'][0][0]


def extract_first_contour(data, levels=[0]):
    """ Extract the contours of a labelised image
        The contours returned are an array of n points of 2 dimension
    """
    ctr = pylab.contour(data, levels=levels, colors='black', origin='image')
    p = ctr.collections[0].get_paths()[0]
    v = p.vertices
    x = v[:,0]
    y = v[:,1]
    contour = np.array([x, y]).T

    # Have contour in a pixel and not between two pixels
    return np.array([np.array(list(map(int, coord))) for coord in contour])


class Image:
    """ name_file
        labels
        contours
        size_x, size_y

        distance
        neighbours : all points at a distance <= self.distance from the contours
        set_scribble0 : points having label0 and being at a distance = self.distance from the contours
        set_scribble1
        rand_scribble : matrix with the scribbles generated
    """

    def __init__(self, name_file, label0=0, segmentation_num=1, delimiter='\t '):
        self.name_file = name_file
        self.label0 = label0
        self.contours = None
        self.distance = None
        self.neighbours = None
        self.set_scribble0 = None
        self.set_scribble1 = None
        self.rand_scribble = None

        if name_file.split('.')[1] == 'txt':
            """ Extract a matrix containing the data of the text file
            Each line of the matrix corresponds to a line on the file
            Each element of a line are separated by "delimiter"
            """
            with open(self.name_file, "r") as f :
                fichier_entier = f.read()
                lines = fichier_entier.split("\n")

            # Separate the lines in integer values
            not_allowed_elem = ['', ' ']  # Elements to remove

            data = [line.split(delimiter) for line in lines]
            for i in range(len(data)):
                data[i] = [int(elem) for elem in data[i] if elem not in not_allowed_elem]

            if [] in data: data.remove([])  # There is sometimes a blank line at the end

            # Reshape the data
            self.labels = np.zeros((len(data), len(data[0])))
            for i in range(len(data)):
                for j in range(len(data[0])):
                    self.labels[i][j] = data[i][j]

        elif name_file.split('.')[1] == 'mat':
            self.labels = load_labels_from_Berkeley_dataset(name_file, segmentation_num)
            str_data = ''
            for i in range(self.labels.shape[1]):
                for j in range(self.labels.shape[0]):
                    str_data += str(int(self.labels[j,i])) + delimiter
                str_data = str_data[:-len(delimiter)] + '\n'

            with open(name_file.split('.')[0] + ".txt", "w") as text_file:
                text_file.write(str_data)

            print("Txt file From Mat saved\n")

        else:
            print("File is neither a texte file nor a matlab file : the program do not know how to manage it")

        # Retreat the values of the labels
        if (np.unique(self.labels) == [1, 2]).all() and self.label0 == 0:
            self.labels -= 1
        elif (np.unique(self.labels) == [0, 1]).all() and self.label0 == 1:
            self.labels += 1

        self.size_y, self.size_x = np.shape(self.labels)

    def show_labels(self):
        plt.matshow(self.labels, origin='upper')

    def extract_contours_labels(self, levels=[0]):
        """ Extract one contours from the labelised image
            (Working for only two classes)
        """
        self.contours = extract_first_contour(self.labels, levels=levels)

    def plot_contours(self, color='black', size=1):
        plt.scatter(self.contours[:,0], self.contours[:, 1], color=color, s=size)

    def set_dist_to_a_point(self, x, y, distance):
        """ Return all the points at a certain Manhattan distance from an input point
            If those points are in the boundary initial image
        """
        man_dist = [[x, y + distance], [x, y - distance],
                    [x + distance, y], [x - distance, y]]
        for dist1 in range(1, distance):
            dist2 = distance - dist1
            man_dist.append([x + dist1, y + dist2])  # right - up
            man_dist.append([x + dist1, y - dist2])  # right - down
            man_dist.append([x - dist1, y + dist2])  # left - up
            man_dist.append([x - dist1, y - dist2])  # left - down

        # Check image size
        i = 0
        while i < len(man_dist):
            if man_dist[i][0] >= self.size_x or man_dist[i][0] < 0 or \
            man_dist[i][1] >= self.size_y or man_dist[i][1] < 0:
                man_dist.remove(man_dist[i])
                i -= 1
            i += 1
        return man_dist

    def neighbours_contours(self, distance):
        """ self.neighours : all the points at a Manhattan distance <= parameter "distance" from the contours
        """

        neighbours = self.set_dist_to_a_point(self.contours[0][0], self.contours[0][1], distance)
        for i in range(1, len(self.contours)):
            new_neighbours = self.set_dist_to_a_point(self.contours[i][0], self.contours[i][1], distance)
            neighbours = np.concatenate((neighbours, new_neighbours), axis=0)

        self.neighbours = np.zeros((self.size_y, self.size_x))
        for point in neighbours:
            self.neighbours[self.size_y - 1 - point[1]][point[0]] = 1

    def plot_neighbours(self):
        plt.matshow(self.neighbours)

    def points_at_a_distance_from_boundary(self, distance):
        """ self.set_scribble0 contains all the points where
                d_manhattan(point, contours) == distance and label(point) == label0
            self.set_scribble1 : idem with the other label
        """

        if self.contours is None:
            self.extract_contours_labels()

        self.distance = distance
        self.neighbours_contours(distance)

        ctr = pylab.contour(self.neighbours, levels=[0], origin='image')
        contour_scribble = []
        self.set_scribble0 = []
        self.set_scribble1 = []

        for cc in ctr.collections[0].get_paths():
            if len(cc.vertices) > 1:
                contour_scribble.append(np.array([np.array(list(map(int, coord))) for coord in cc.vertices]))

        for contour in contour_scribble:
            if self.labels[self.size_y -1 - contour[0, 1]][contour[0, 0]] == self.label0:
                if self.set_scribble0 == []:
                    self.set_scribble0 = contour
                else:
                    self.set_scribble0 = np.concatenate((self.set_scribble0, contour))
            else:
                if self.set_scribble1 == []:
                    self.set_scribble1 = contour
                else:
                    self.set_scribble1 = np.concatenate((self.set_scribble1, contour))

    def plot_set_scribbles(self):
        plt.scatter(self.set_scribble0[:,0], self.set_scribble0[:,1], s=1)
        plt.scatter(self.set_scribble1[:,0], self.set_scribble1[:,1], s=1)

    def generate_scribble_at_distance(self, distance, nb_scribbles):
        if self.contours is None:
            print("Computing contours form labels...")
            self.extract_contours_labels()

        self.distance = distance
        print("Computing {} points at the distance {}...".format(nb_scribbles, self.distance))
        self.points_at_a_distance_from_boundary(self.distance)

        print("Extracting scribbles...")
        self.rand_scribble = np.zeros((self.size_y, self.size_x)) - 1
        for i in range(nb_scribbles):
            # Label 0
            n = random.randint(0, len(self.set_scribble0) - 1)
            self.rand_scribble[self.size_y -1 - self.set_scribble0[n, 1], self.set_scribble0[n, 0]] = self.label0

            # Label 1
            m = random.randint(0, len(self.set_scribble1) - 1)
            self.rand_scribble[self.size_y -1 - self.set_scribble1[m, 1], self.set_scribble1[m, 0]] = self.label0 + 1

    def show_scribbles(self):
        plt.matshow(self.rand_scribble)
        plt.title("Scribbles generated")

    def save_scribble_txt(self, name_file, delimiter='\t '):
        str_data = ''
        for i in range(self.size_y):
            for j in range(self.size_x):
                str_data += str(int(self.rand_scribble[i,j])) + delimiter
            str_data = str_data[:-len(delimiter)] + '\n'

        with open(name_file, "w") as text_file:
            text_file.write(str_data)
        print("Txt file saved")

    def generate_multi_scribbles_and_save(self, distance_list, nb_points_list, K, name_file, delimiter='\t '):
        if '.txt' in name_file:
            name_file = name_file[:-4]
        all_files_names = ''

        for d in distance_list:
            for nb_points in nb_points_list:
                for k in range(K):
                    self.generate_scribble_at_distance(d, nb_points)
                    name = name_file + 'd_' + str(d) + '_n_' + str(nb_points) + '_' + str(k) + '.txt'
                    self.save_scribble_txt(name, delimiter=delimiter)


def main(argv):
    arg_dict = {}
    switches = {'labelsFile':str,
        'distanceList':list,
        'nbPointList':list,
        'svgFileName':str}
    singles = ''.join([x[0]+':' for x in switches])
    long_form = [x+'=' for x in switches]
    d = {x[0]+':':'--'+x for x in switches}
    try:
        opts, args = getopt.getopt(argv, singles, long_form)
    except getopt.GetoptError:
        print("bad arg")
        sys.exit(2)

    for opt, arg in opts:
        if opt[1]+':' in d: o=d[opt[1]+':'][2:]
        elif opt in d.values(): o=opt[2:]
        else: o = ''
        #print(opt, arg, o)
        if o and arg:
            arg_dict[o]=ast.literal_eval(arg)

        if not o or not isinstance(arg_dict[o], switches[o]):
            print(opt, arg, " Error: bad arg")
            sys.exit(2)

    labelsFile = arg_dict["labelsFile"]
    distanceList = arg_dict["distanceList"]
    nbPointList = arg_dict["nbPointList"]
    svgFileName = arg_dict["svgFileName"]

    truth = Image(labelsFile);
    truth.generate_multi_scribbles_and_save(distanceList, nbPointList, 1, svgFileName)
    print("Finished")


if __name__ == '__main__':
    #4 arguments : "labels_croco.txt" liste_des_distances, liste_des_nb_de_points, nom_fichier_svg
    #print("You are in the Python Script")
    main(sys.argv[1:])
