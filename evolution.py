import tensorflow as tf
import numpy as np
import random

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

from Network import MLP, MLPDescriptor, initializations, activations, TCNN, CNN, ConvDescriptor, TConvDescriptor
from Mutation import MLP_Mutation, CNN_Mutation, TCNN_Mutation
from auxiliary_functions import batch
from metrics import mse, accuracy_error
import os

from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.models import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

descs = {"ConvDescriptor": CNN, "MLPDescriptor": MLP, "TConvDescriptor": TCNN}

# This function set the seeds of the tensorflow function
# to make this notebook's output stable across runs


#####################################################################################################

class MyContainer(object):
    # This class does not require the fitness attribute
    # it will be  added later by the creator
    def __init__(self, descriptor_list):
        # Some initialisation with received values
        print(descriptor_list)
        self.descriptor_list = descriptor_list


class Evolving:
    def __init__(self, loss="XEntropy", desc_list=(MLPDescriptor, ), compl=False, 
                 x_trains=None, y_trains=None, x_tests=None, y_tests=None, 
                 evaluation="Accuracy_error", n_inputs=((28, 28),), n_outputs=((10,),), 
                 batch_size=100, population=20, generations=20, iters=10, lrate=0.01, sel=0,
                 n_layers=10, max_layer_size=100, max_filter=4, max_stride=3, seed=0, cxp=0, 
                 mtp=1, no_dropout=False, no_batch_norm=False, evol_kwargs={}, sel_kwargs={}, 
                 ev_alg=1, hyperparameters={}, custom_mutations={}, add_obj=0):
        """
        This is the main class in charge of evolving model descriptors.
        """

        np.random.seed(seed)
        #tf.random.set_random_seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)

        self.n_inputs = n_inputs                                        # List of lists with the dimensions of the inputs of each network
        self.n_outputs = n_outputs                                      # List of lists with the dimensions of the outputs of each network

        self.network_descriptor = {}                                    # dict {"net_id": net_desc}
        self.nlayers = n_layers                                         # Maximum number of layers
        self.max_lay = max_layer_size                                   # Maximum number of neurons per layer
        self.max_filter = max_filter                                    # Maximum size of filter
        self.max_stride = max_stride                                    # Maximum stride
        self.descriptors = desc_list                                    # Number of MLPs in the model
        self.loss_function = None                                       # Loss function to be used for training the model
        self.evaluation = None                                          # Function for evaluating the model
        self.define_loss_eval(loss, evaluation)                         # Assigning the values of the previous variables

        self.batch_size = batch_size                                    # Batch size for training
        self.predictions = {}                                           # dict {"Net_id": net_output}
        self.lrate = lrate                                              # Learning rate
        self.iters = iters                                              # Number of training epochs
        self.train_inputs = {}                                          # Training data (X)
        self.train_outputs = {}                                         # Training data (y)
        self.test_inputs = {}                                           # Test data (X)
        self.test_outputs = {}                                          # Test data (y)
        self.data_save(x_trains, y_trains, x_tests, y_tests)            # Save data in the previous dicts
        self.complex = self.is_complex(compl, loss, evaluation, hyperparameters)
        
        self.toolbox = base.Toolbox()
        self.ev_alg = None                                              # DEAP evolutionary algorithm function
        self.cXp = cxp if len(desc_list) > 1 else 0                     # Crossover probability. 0 in the simple case
        self.mtp = mtp if len(desc_list) > 1 else 1                     # Mutation probability. 1 in the simple case
        self.evol_kwargs = {}                                           # Parameters for the main DEAP function
        self.evol_function = ev_alg                                     # main DEAP function index
        self.generations = generations                                  # Number of generations
        self.population_size = population                               # Individuals in a population
        self.ev_hypers = hyperparameters                                # Hyperparameters to be evolved (e.g., optimizer, batch size)
        self.initialize_deap(sel, sel_kwargs, ev_alg, evol_kwargs, 
                             no_batch_norm, no_dropout, custom_mutations, add_obj)      # Initialize DEAP-related matters
        
    def data_save(self, x_trains, y_trains, x_tests, y_tests):
        """
        Filling the dicts with the training data
        :param x_trains: X data for training
        :param y_trains: y data for training
        :param x_tests: X data for testing
        :param y_tests: y data for testing
        :return: --
        """

        if isinstance(x_trains, dict):
            self.train_inputs = x_trains
            self.train_outputs = y_trains

            self.test_inputs = x_tests
            self.test_outputs = y_tests
        else:
            for i, x in enumerate(x_trains):
                self.train_inputs["i" + str(i)] = x
                self.train_outputs["o" + str(i)] = y_trains[i]

            for i, x in enumerate(x_tests):
                self.test_inputs["i" + str(i)] = x
                self.test_outputs["o" + str(i)] = y_tests[i]

    def define_loss_eval(self, loss, evaluation):
        """
        Define the loss and evaluation function. Writing the (string) names of the predefined functions is accepted.
        :param loss: Loss function. Either string (predefined) or customized by the user.
        :param evaluation: Evaluation function. Either string (predefined) or customized by the user.
        :return:
        """

        #losses = {"MSE": tf.losses.mean_squared_error, "XEntropy": tf.losses.softmax_cross_entropy}
        losses = {"MSE": tf.losses.mean_squared_error, "XEntropy": tf.compat.v1.losses.softmax_cross_entropy}       
        evals = {"MSE": mse, "Accuracy_error": accuracy_error}

        if type(loss) is str:
            self.loss_function = losses[loss]
        else:
            self.loss_function = loss

        if type(evaluation) is str:
            self.evaluation = evals[evaluation]
        else:
            self.evaluation = evaluation
    
    def is_complex(self, compl, loss, evaluation, hyperparameters):
        if compl:
            return True
        elif (type(loss) is not str) or (type(evaluation) is not str):
            return True
        elif self.descriptors[0] is not MLPDescriptor:
            return True
        elif len(self.descriptors) > 1 or len(hyperparameters) > 0:
            return True
        
        return False
        
    
    def initialize_deap(self, sel, sel_kwargs, ev_alg, ev_kwargs, no_batch, no_drop, custom_mutations, add_obj):
        """
        Initialize DEAP algorithm
        :param sel: Selection method
        :param sel_kwargs: Hyperparameters for the selection methods, e.g., size of the tournament if that method is selected
        :param ev_alg: DEAP evolutionary algorithm (EA)
        :param ev_kwargs: Hyperparameters for the EA, e.g., mutation or crossover probability.
        :param no_batch: Whether the evolutive process includes batch normalization in the networks or not
        :param no_drop: Whether the evolutive process includes dropout in the networks or not
        :param add_obj: Number of additional objectives
        :return: --
        """

        deap_algs = [algorithms.eaSimple, algorithms.eaMuPlusLambda, algorithms.eaMuCommaLambda, algorithms.eaGenerateUpdate]

        creator.create("Fitness", base.Fitness, weights=[-1.0]*(len(self.test_outputs) + add_obj))

        creator.create("Individual", MyContainer, fitness=creator.Fitness)

        self.toolbox.register("individual", self.init_individual, creator.Individual, no_batch, no_drop)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self.eval_individual)
        self.toolbox.register("mate", cross, creator.Individual)
        self.toolbox.register("mutate", mutations, self.ev_hypers, self.max_lay, no_batch, no_drop, custom_mutations)

        sel_methods = [tools.selBest, tools.selTournament, tools.selNSGA2]

        self.toolbox.register("select", sel_methods[sel], **sel_kwargs)

        if len(ev_kwargs) == 0:
            if ev_alg == 0:
                self.evol_kwargs = {"cxpb": self.cXp, "mutpb": self.mtp, "ngen": self.generations, "verbose": 1}
            if ev_alg == 1:
                self.evol_kwargs = {"mu": self.population_size, "lambda_": self.population_size, "cxpb": self.cXp, "mutpb": self.mtp, "ngen": self.generations, "verbose": 1}
            if ev_alg == 2:
                self.evol_kwargs = {"mu": self.population_size, "lambda_": self.population_size, "cxpb": self.cXp, "mutpb": self.mtp, "ngen": self.generations, "verbose": 1}

        self.ev_alg = deap_algs[ev_alg]

    def evolve(self):
        """
        Actual evolution of individuals
        :return: The last generation, a log book (stats) and the hall of fame (the best individuals found)
        """

        pop = self.toolbox.population(n=self.population_size)
        hall_of = tools.HallOfFame(self.population_size)
        stats = tools.Statistics(lambda ind: ind.fitness.values)

        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)
        result, log_book = self.ev_alg(pop, self.toolbox, **self.evol_kwargs, stats=stats, halloffame=hall_of)

        return result, log_book, hall_of

    def init_individual(self, init_ind, no_batch, no_drop):
        """
        Creation of a single individual
        :param init_ind: DEAP function for transforming a network descriptor, or a list of descriptors + evolvable hyperparameters into a DEAP individual
        :param no_batch: Boolean, whether batch normalization is included into the evolution or not
        :param no_drop: Boolean, whether dropout is included into the evolution or not
        :return: a DEAP individual
        """

        network_descriptor = {}

        if not self.complex:  # Simple case
            network_descriptor["n0"] = MLPDescriptor()
            network_descriptor["n0"].random_init(self.train_inputs["i0"].shape[1:], self.train_outputs["o0"].shape[1], self.nlayers, self.max_lay, None, None, no_drop, no_batch)
        else:  # Custom case
            for i, descriptor in enumerate(self.descriptors):
                network_descriptor["n" + str(i)] = descriptor()
                network_descriptor["n" + str(i)].random_init(self.n_inputs[i], self.n_outputs[i], self.nlayers, self.max_lay, self.max_stride, self.max_filter, no_drop, no_batch)
        network_descriptor["hypers"] = {}
        if len(self.ev_hypers) > 0:

            for hyper in self.ev_hypers:
                network_descriptor["hypers"][hyper] = np.random.choice(self.ev_hypers[hyper])

        return init_ind(network_descriptor)

    def eval_individual(self, individual):
        """
        Function for evaluating an individual.
        :param individual: DEAP individual
        :return: Fitness value.
        """

        if not self.complex:
            ev = self.single_net_eval(individual)
        else:
            ev = self.eval_multinetwork(individual)
        return ev

    def single_net_eval(self, individual):
        """
        Function for evolving a single individual. No need of the user providing a evaluation function
        :param individual: DEAP individual
        :return: Fitness value
        """
        net = MLP(individual.descriptor_list["n0"])
        
        inp = Input(shape=self.n_inputs[0])
        out = Flatten()(inp)
        out = net.building(out)
        model = Model(inputs=inp, outputs=out)
        
        opt = tf.keras.optimizers.Adam(learning_rate=self.lrate)
        model.compile(loss=self.loss_function, optimizer=opt, metrics=[])
        
        model.fit(self.train_inputs['i0'], self.train_outputs['o0'], epochs=self.iters, batch_size=self.batch_size, verbose=0)
        
        #model.summary()
        
        ev = model.evaluate(self.test_inputs['i0'], self.test_outputs['o0'], verbose=0)
        
        if isinstance(ev, float):
            ev = (ev,)
            
        return ev

    def eval_multinetwork(self, individual):
        """
        Function for evaluating a DEAP individual consisting of more than a single MLP. The user must have implemented the
        training and evaluation functions.
        :param individual: DEAP individual
        :return: Fitness value
        """
        nets = {}
    
        for index, net in enumerate(individual.descriptor_list.keys()):
            if "hypers" not in net:
                nets[net] = descs[self.descriptors[index].__name__](individual.descriptor_list[net])

        models = self.loss_function(nets, self.train_inputs, self.train_outputs, self.batch_size, individual.descriptor_list["hypers"])

        ev = self.evaluation(models, self.test_inputs, self.test_outputs, individual.descriptor_list["hypers"])

        return ev


def mutations(ev_hypers, max_lay, batch_normalization, drop, custom_mutations, individual):
    """
    Mutation operators for individuals. They can affect any network or the hyperparameters.
    :param ev_hypers: Hyperparameters not included in the networks to be evolved
    :param max_lay: Maximum number of layers in networks
    :param batch_normalization: Whether batch normalization is part of the evolution or not
    :param drop:Whether dropout is part of the evolution or not
    :param individual: DEAP individual. Contains a dict where the keys are the components of the model
    :return: Mutated version of the DEAP individual.
    """
    
    mutation_types = {'MLPDescriptor': MLP_Mutation, 'ConvDescriptor': CNN_Mutation, 'TConvDescriptor': TCNN_Mutation}

    nets = list(individual.descriptor_list.keys())
    nets.remove("hypers")

    network = individual.descriptor_list[np.random.choice(nets)]
    
    network_mutation = mutation_types[network.__class__.__name__](ev_hypers, max_lay, batch_normalization, drop, individual)
        
    if not custom_mutations:
        network_custom_mutations = [] # If no custom mutations are passed, each network's mutations will be applied
    else:
        network_custom_mutations = custom_mutations[network.__class__.__name__]
    
    network_mutation.apply_random_mutation(network, network_custom_mutations)
    
    return individual,

def cross(init_ind, ind1, ind2):
    """
    Crossover operator for individuals. Cannot be applied in the simple case, as it randomly interchanges model components
    :param init_ind: DEAP function for initializing dicts (in this case) as DEAP individuals
    :param ind1: 1st individual to be crossed
    :param ind2: 2st individual to be crossed
    :return: Two new DEAP individuals, the crossed versions of the incoming parameters.
    """

    keys = list(ind1.descriptor_list.keys())
    # We randomly select the keys of the components that will be interchanged.
    cx_point = np.random.choice(keys, size=np.random.randint(1, len(keys)) if len(keys) > 2 else 1, replace=False)
    new1 = {}
    new2 = {}

    for key in keys:
        if key in cx_point:
            new1[key] = ind1.descriptor_list[key]
            new2[key] = ind2.descriptor_list[key]
        else:
            new1[key] = ind2.descriptor_list[key]
            new2[key] = ind1.descriptor_list[key]

    return init_ind(new1), init_ind(new2)
