import math
import nn


import util
###########################################################################
class NaiveBayesDigitClassificationModel(object):

    def __init__(self):
        self.conditionalProb = None
        self.prior = None
        self.features = None
        self.k = 1 # this is the smoothing parameter, ** use it in your train method **
        self.automaticTuning = True # Look at this flag to decide whether to choose k automatically ** use this in your train method **
        self.legalLabels = range(10)

    def train(self, dataset):
        # this is a list of all features in the training set.
        self.features = list(set([f for datum in dataset.trainingData for f in datum.keys()]))

        kgrid = [0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.5, 1, 5]
        self.trainAndTune(dataset, kgrid)

    def trainAndTune(self, dataset, kgrid):
        """
        Trains the classifier by collecting counts over the training data, and
        stores the Laplace smoothed estimates so that they can be used to classify.
        Evaluate each value of k in kgrid to choose the smoothing parameter
        that gives the best accuracy on the held-out validationData.

        trainingData and validationData are lists of feature Counters. The corresponding
        label lists contain the correct label for each datum.

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """
        trainingData = dataset.trainingData
        trainingLabels = dataset.trainingLabels
        validationData = dataset.validationData
        validationLabels = dataset.validationLabels

        bestAccuracyCount = -1  # best accuracy so far on validation set
        #Common training - get all counts from training data
        #We only do it once - save computation in tuning smoothing parameter
        commonPrior = util.Counter()  # Prior probability over labels
        commonConditionalProb = util.Counter()  #Conditional probability of feature feat being 1 indexed by (feat, label)
        commonCounts = util.Counter()  #how many time I have seen feature 'feat' with label 'y' whether inactive or active
        bestParams = (commonPrior, commonConditionalProb, kgrid[0])  # used for smoothing part  trying various Laplace factors kgrid

        for i in range(len(trainingData)):
            datum = trainingData[i]
            label = int(trainingLabels[i])
            "*** YOUR CODE HERE to complete populating commonPrior, commonCounts, and commonConditionalProb ***"
            commonPrior[label] += 1
            for feat, value in datum.items():
                commonCounts[(feat, label)] += 1 # add 1 for num times ive seen the feature
                if value > 0:
                    commonConditionalProb[(feat, label)] += 1
            # commonConditionalProb = ??
            """
            P(Y|f_1,...,f_n) = P(Y)P(f_1|Y)...P(f_n|Y) / P(f_1)...P(f_2)
            """
            #util.raiseNotDefined()

        for k in kgrid:  # smoothing parameter tuning loop
            prior = util.Counter()
            conditionalProb = util.Counter()
            counts = util.Counter()

            # get counts from common training step
            for key, val in commonPrior.items():
                prior[key] += val
            for key, val in commonCounts.items():
                counts[key] += val
            for key, val in commonConditionalProb.items():
                conditionalProb[key] += val

            # smoothing:
            for label in self.legalLabels:
                for feat in self.features:
                    "*** YOUR CODE HERE to update conditionalProb and counts using Lablace smoothing ***"
                    conditionalProb[(feat, label)] += k
                    counts[(feat, label)] += k * len(self.legalLabels)
                    #util.raiseNotDefined()

            # normalizing:
            prior.normalize()
            "**** YOUR CODE HERE to normalize conditionalProb "
            for x, count in conditionalProb.items():
                conditionalProb[x] = count / counts[x]


            self.prior = prior
            self.conditionalProb = conditionalProb

            # evaluating performance on validation
            predictions = self.classify(validationData)
            accuracyCount = [predictions[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)

            print("Performance on validation set for k=%f: (%.1f%%)" % (
            k, 100.0 * accuracyCount / len(validationLabels)))
            if accuracyCount > bestAccuracyCount:
                bestParams = (prior, conditionalProb, k)
                bestAccuracyCount = accuracyCount
            # end of automatic tuning loop

        self.prior, self.conditionalProb, self.k = bestParams
        print("Best Performance on validation set for k=%f: (%.1f%%)" % (
            self.k, 100.0 * bestAccuracyCount / len(validationLabels)))


    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.
        You shouldn't modify this method.
        """
        guesses = []
        self.posteriors = [] # Log posteriors are stored for later data analysis
        for datum in testData:
            ("***YOUR CODE HERE***  use calculateLogJointProbabilities() to compute posterior per datum  and use"
             "it to find best guess digit for datum and at the end accumulate in self.posteriors for later use")
            pos = self.calculateLogJointProbabilities(datum)
            guesses.append(pos.argMax())
            self.posteriors.append(pos)
            #util.raiseNotDefined()

        return guesses

    def calculateLogJointProbabilities(self, datum):
        """
        Returns the log-joint distribution over legal labels and the datum.
        Each log-probability should be stored in the log-joint counter, e.g.
        logJoint[3] = <Estimate of log( P(Label = 3, datum) )>

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """
        logJoint = util.Counter()

        for label in self.legalLabels:
            "*** YOUR CODE HERE, to populate logJoint() list ***"
            logJoint[label] = math.log(self.prior[label])
            for feat, val in datum.items():
                if val > 0:
                    # use stored probability if feature is active
                    logJoint[label] += math.log(self.conditionalProb[(feat, label)])
                else:
                    # use 1 - prob if feature is inactive
                    logJoint[label] += math.log(1 - self.conditionalProb[(feat, label)])
            #util.raiseNotDefined()
        return logJoint

################################################################################3
class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x, self.w)
        #util.raiseNotDefined()

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        return 1 if nn.as_scalar(self.run(x)) >= 0 else -1
        #util.raiseNotDefined()

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        flag = 1
        while flag == 1:
            flag = 0
            for batch_x, batch_y in dataset.iterate_once(batch_size=1):
                p = self.get_prediction(batch_x)
                if p != nn.as_scalar(batch_y):
                    flag = 1
                    nn.Parameter.update(self.w, batch_x, nn.as_scalar(batch_y))
        #util.raiseNotDefined()

########################################################################33
class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here. Here you setup the architecture of your NN, meaning how many
        # layers and corresponding weights, what is the batch_size, and learning_rate.
        "*** YOUR CODE HERE ***"
        """
        layers/weights
        parameter matrices W1 and W2, parameter vectors b1, b2
        W1 is i x h mat, where i = input height, h = hidden layer size
        h1 = f_1(x) = relu(x . W1 + b1)
        h2 = f_2(h1) = relu(h1 . W2 + b2)
        y' = f_3(h2) = h2 . W3 + b3
        start w one layer!
        mse loss as loss function
        
        input: one real number
        output: another real number
        
        in general, num rows = num cols of prev, num cols = num inputs of second
        activation of last layer is the output
        """


        # 1 layer network
        """
        param = 100, batch = 50, learn = 0.01 -> Your final loss is: 0.001000 in 1,328,7800 number of processing iterations
        param = 100, batch = 20, learn = 0.01 -> Your final loss is: 0.001000 in 9,725,600 number of processing iterations
        param = 100, batch = 20, learn = 0.02 -> Your final loss is: 0.000995 in 3,256,600 number of processing iterations

        
        
        """
        h = 200
        self.w1 = nn.Parameter(1, h)
        self.b1 = nn.Parameter(1, h)
        self.w2 = nn.Parameter(h, 1)
        self.b2 = nn.Parameter(1, 1) # the result
        self.batch_size = 100
        self.learning_rate = 0.1
        self.threshold = 0.01
        #util.raiseNotDefined()


    def run(self, x):
        """
        Runs the model for a batch of examples.
        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        xW1_b1 = nn.AddBias(nn.Linear(x, self.w1), self.b1)
        A1 = nn.ReLU(xW1_b1)
        return nn.AddBias(nn.Linear(A1, self.w2), self.b2)
        #util.raiseNotDefined()


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)
        #util.raiseNotDefined()

    def train(self, dataset):
        """
            Trains the model.
        """
        "*** YOUR CODE HERE ***"
        loss = 1
        while loss > self.threshold:
            for batch_x, batch_y in dataset.iterate_once(self.batch_size):
                loss_object = self.get_loss(batch_x, batch_y)
                grad_w1, grad_b1, grad_w2, grad_b2 = nn.gradients(loss_object, [self.w1, self.b1, self.w2, self.b2])
                self.b1.update(grad_b1, -self.learning_rate)
                self.b2.update(grad_b2, -self.learning_rate)
                self.w1.update(grad_w1, -self.learning_rate)
                self.w2.update(grad_w2, -self.learning_rate)
            loss = nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y)))
            #print("Your final loss is: %f in %d number of processing iterations" % (loss, dataset.iterations))
        #util.raiseNotDefined()

##########################################################################
class DigitClassificationModel(object):
    """
    A second model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to classify each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        h = 100
        self.w1 = nn.Parameter(784, h)
        self.b1 = nn.Parameter(1, h)

        self.w2 = nn.Parameter(h, h)
        self.b2 = nn.Parameter(1, h)

        self.w3 = nn.Parameter(h, 10)
        self.b3 = nn.Parameter(1, 10)

        self.learning_rate = 0.1
        self.batch_size = 100
        self.threshold = 0.975
        #util.raiseNotDefined()

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        A1 = nn.AddBias(nn.Linear(x, self.w1), self.b1)
        Z1 = nn.ReLU(A1)
        A2 = nn.AddBias(nn.Linear(Z1, self.w2), self.b2)
        Z2 = nn.ReLU(A2)
        return nn.AddBias(nn.Linear(Z2, self.w3), self.b3)
        #util.raiseNotDefined()

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)
        #util.raiseNotDefined()

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        validation_acc = 0.0
        while validation_acc < self.threshold:
            # iterate over the data ONCE
            for batch_x, batch_y in dataset.iterate_once(self.batch_size):
                soft_max_loss = self.get_loss(batch_x, batch_y)
                grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3 = nn.gradients(soft_max_loss, [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3])
                self.b1.update(grad_b1, -self.learning_rate)
                self.b2.update(grad_b2, -self.learning_rate)
                self.b3.update(grad_b3, -self.learning_rate)
                self.w1.update(grad_w1, -self.learning_rate)
                self.w2.update(grad_w2, -self.learning_rate)
                self.w3.update(grad_w3, -self.learning_rate)
            validation_acc = dataset.get_validation_accuracy()
        print('stopped training with validation accuracy:', validation_acc, 'and epoch', dataset.epoch)
        #util.raiseNotDefined()

###################################################################################
class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.threshold = 0.85

        # tweakable parameters
        h = 50
        self.learning_rate = 0.05
        self.batch_size = 50

        # input layer
        self.w_initial = nn.Parameter(47, h)
        self.b_initial = nn.Parameter(1, h)

        # hidden layer
        self.w_hidden = nn.Parameter(h, h)
        self.b_hidden = nn.Parameter(1, h)

        # output layer
        self.w_output = nn.Parameter(h, len(self.languages)) 
        self.b_output = nn.Parameter(1, len(self.languages))
        # it has to output to one of these languages
        #util.raiseNotDefined()


    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the initial (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        a0 = nn.AddBias(nn.Linear(xs[0], self.w_initial), self.b_initial)
        z0 = nn.ReLU(a0) # initial
        a1 = nn.AddBias(nn.Linear(z0, self.w_hidden), self.b_hidden)
        ht = nn.ReLU(a1)
        for i in range(1, len(xs)):
            # h is previous timestep's calculation
            w = nn.Linear(xs[i], self.w_initial)
            wh = nn.AddBias(nn.Linear(ht, self.w_hidden), self.b_hidden) # i am not sure this will work
            ht = nn.ReLU(nn.Add(w, wh))
        return nn.AddBias(nn.Linear(ht, self.w_output), self.b_output)
        #util.raiseNotDefined()

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs), y)
        #util.raiseNotDefined()

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        "*** hint: User get_validation_accuracy() to decide when to finish learning ***"
        validation_acc = 0.0
        while validation_acc < self.threshold:
            # iterate over the data ONCE
            for batch_x, batch_y in dataset.iterate_once(self.batch_size):
                soft_max_loss = self.get_loss(batch_x, batch_y)
                grad_wi, grad_bi, grad_wh, grad_bh, grad_wo, grad_bo = nn.gradients(soft_max_loss, [self.w_initial, self.b_initial, self.w_hidden, self.b_hidden, self.w_output, self.b_output])
                self.b_initial.update(grad_bi, -self.learning_rate)
                self.b_hidden.update(grad_bh, -self.learning_rate)
                self.b_output.update(grad_bo, -self.learning_rate)
                self.w_initial.update(grad_wi, -self.learning_rate)
                self.w_hidden.update(grad_wh, -self.learning_rate)
                self.w_output.update(grad_wo, -self.learning_rate)
            validation_acc = dataset.get_validation_accuracy()
        print('stopped training with validation accuracy:', validation_acc, 'and epoch', dataset.epoch)
        #util.raiseNotDefined()
