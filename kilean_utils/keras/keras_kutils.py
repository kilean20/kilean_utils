from tensorflow import keras


def cluster_acc(Y_pred, Y):
  from sklearn.utils.linear_assignment_ import linear_assignment
  assert Y_pred.size == Y.size
  D = max(Y_pred.max(), Y.max())+1
  w = np.zeros((D,D), dtype=np.int64)
  for i in range(Y_pred.size):
    w[Y_pred[i], Y[i]] += 1
  ind = linear_assignment(w.max() - w)
  return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w


class VAE_latent(keras.layers)


class VAE(torch.nn.Module):
    def __init__(self,node_list,activation):
        super(VAE, self).__init__()

        # self.node_list = node_list
        # self.activation = activation

        self.encoder = []
        for i in range(len(node_list)-2):
            self.encoder.append(torch.nn.Linear(node_list[i],node_list[i+1]))
            self.encoder.append(activation)
        self.encoder = torch.nn.Sequential(*self.encoder)
        self.Mu = torch.nn.Linear(node_list[-2],node_list[-1])
        self.LogVar = torch.nn.Linear(node_list[-2],node_list[-1])
        torch.nn.init.zeros_(self.LogVar.weight)
        torch.nn.init.zeros_(self.LogVar.bias)


        self.decoder = []
        for i in range(len(node_list)-1,0,-1):
            self.decoder.append(torch.nn.Linear(node_list[i],node_list[i-1]))
            self.decoder.append(activation)
        self.decoder = torch.nn.Sequential(*self.decoder)


    def reparameterize(self, mu: Variable, logvar: Variable) -> Variable:
        """THE REPARAMETERIZATION IDEA:

        For each training sample (we get 128 batched at a time)

        - take the current learned mu, stddev for each of the ZDIMS
          dimensions and draw a random sample from that distribution
        - the whole network is trained so that these randomly drawn
          samples decode to output that looks like the input
        - which will mean that the std, mu will be learned
          *distributions* that correctly encode the inputs
        - due to the additional KLD term (see loss_function() below)
          the distribution will tend to unit Gaussians

        Parameters
        ----------
        mu : [128, ZDIMS] mean matrix
        logvar : [128, ZDIMS] variance matrix

        Returns
        -------

        During training random sample from the learned ZDIMS-dimensional
        normal distribution; during inference its mean.

        """

        if self.training:
            # multiply log variance with 0.5, then in-place exponent
            # yielding the standard deviation
            std = logvar.mul(0.5).exp_()  # type: Variable
            # - std.data is the [128,ZDIMS] tensor that is wrapped by std
            # - so eps is [128,ZDIMS] with all elements drawn from a mean 0
            #   and stddev 1 normal distribution that is 128 samples
            #   of random ZDIMS-float vectors
            eps = Variable(std.data.new(std.size()).normal_())
            # - sample from a normal distribution with standard
            #   deviation = std and mean = mu by multiplying mean 0
            #   stddev 1 sample with desired std and mu, see
            #   https://stats.stackexchange.com/a/16338
            # - so we have 128 sets (the batch) of random ZDIMS-float
            #   vectors sampled from normal distribution with learned
            #   std and mu for the current input
            return eps.mul(std).add_(mu)

        else:
            # During inference, we simply spit out the mean of the
            # learned distribution for the current input.  We could
            # use a random sample from the distribution, but mu of
            # course has the highest probability.
            return mu


    def forward(self, x: Variable) -> (Variable, Variable, Variable):
        x = self.encoder(x)
        mu = torch.clamp(self.Mu(x), min=-1000.0,max=1000.0)
        logvar = torch.clamp(self.LogVar(x), min=-10.0,max=10.0)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
    