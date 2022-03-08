import datetime
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow_probability as tfp
import tensorflow.keras.layers as tfkl
tfd,tfpl = tfp.distributions,tfp.layers
import tensorflow.keras.backend as tfkb
from tensorflow.keras.callbacks import Callback
from sklearn.preprocessing import StandardScaler

class Evaluator(object):
    def __init__(self, y, t, y_cf=None, mu0=None, mu1=None):
        self.y = y
        self.t = t
        self.y_cf = y_cf
        self.mu0 = mu0
        self.mu1 = mu1
        if mu0 is not None and mu1 is not None:
            self.true_ite = mu1 - mu0

    def rmse_ite(self, ypred1, ypred0):
        idx1, idx0 = self.t, 1-self.t
        ite1, ite0 = (self.y - ypred0) * idx1, (ypred1 - self.y)*idx0
        pred_ite = ite1 + ite0
        return tf.math.sqrt(tfkb.mean(tf.math.square(self.true_ite - pred_ite)))

    def abs_ate(self, ypred1, ypred0):
        return tf.math.abs(tfkb.mean(ypred1 - ypred0) - tfkb.mean(self.true_ite))

    def pehe(self, ypred1, ypred0):
        return tf.math.sqrt(tfkb.mean(tf.math.square((self.mu1 - self.mu0) - (ypred1 - ypred0))))

    def y_errors(self, y0, y1):
        ypred = (1 - self.t) * y0 + self.t * y1
        ypred_cf = self.t * y0 + (1 - self.t) * y1
        return self.y_errors_pcf(ypred, ypred_cf)

    def y_errors_pcf(self, ypred, ypred_cf):
        rmse_factual = tf.math.sqrt(tfkb.mean(tf.math.square(ypred - self.y)))
        rmse_cfactual = tf.math.sqrt(tfkb.mean(tf.math.square(ypred_cf - self.y_cf)))
        return rmse_factual, rmse_cfactual

    def calc_stats(self, ypred1, ypred0):
        ite = self.rmse_ite(ypred1, ypred0)
        ate = self.abs_ate(ypred1, ypred0)
        pehe = self.pehe(ypred1, ypred0)
        return ite, ate, pehe

def pdist2sq(A, B):
    #helper for PEHEnn
    #calculates squared euclidean distance between rows of two matrices  
    #https://gist.github.com/mbsariyildiz/34cdc26afb630e8cae079048eef91865
    # squared norms of each row in A and B
    na = tf.reduce_sum(tf.square(A), 1)
    nb = tf.reduce_sum(tf.square(B), 1)    
    # na as a row and nb as a column vectors
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])
    # return pairwise euclidean difference matrix
    D=tf.reduce_sum((tf.expand_dims(A, 1)-tf.expand_dims(B, 0))**2,2) 
    return D

#https://towardsdatascience.com/implementing-macro-f1-score-in-keras-what-not-to-do-e9f1aa04029d
class Full_Metrics(Callback):
    def __init__(self,data, verbose=0):   
        super(Full_Metrics, self).__init__()
        self.data=data #feed the callback the full dataset
        self.verbose=verbose

        #needed for PEHEnn; Called in self.find_ynn
        self.data['o_idx']=tf.range(self.data['t'].shape[0])
        self.data['c_idx']=self.data['o_idx'][self.data['t'].squeeze()==0] #These are the indices of the control units
        self.data['t_idx']=self.data['o_idx'][self.data['t'].squeeze()==1] #These are the indices of the treated units
    
    def split_pred(self,concat_pred):
        #this helps us keep ptrack of things so we don't make mistakes
        preds={}
        preds['y0_pred'] = self.data['y_scaler'].inverse_transform(np.reshape(concat_pred[:, 0],[-1,1]))[:,0]
        preds['y1_pred'] = self.data['y_scaler'].inverse_transform(np.reshape(concat_pred[:, 1],[-1,1]))[:,0]
        preds['phi'] = concat_pred[:, 2:]
        return preds

    def find_ynn(self, Phi):
        #helper for PEHEnn
        PhiC, PhiT =tf.dynamic_partition(Phi,tf.cast(tf.squeeze(self.data['t']),tf.int32),2) #separate control and treated reps
        dists=tf.sqrt(pdist2sq(PhiC,PhiT)) #calculate squared distance then sqrt to get euclidean
        yT_nn_idx=tf.gather(self.data['c_idx'],tf.argmin(dists,axis=0),1) #get c_idxs of smallest distances for treated units
        yC_nn_idx=tf.gather(self.data['t_idx'],tf.argmin(dists,axis=1),1) #get t_idxs of smallest distances for control units
        yT_nn=tf.gather(self.data['y'],yT_nn_idx,1) #now use these to retrieve y values
        yC_nn=tf.gather(self.data['y'],yC_nn_idx,1)
        y_nn=tf.dynamic_stitch([self.data['t_idx'],self.data['c_idx']],[yT_nn,yC_nn]) #stitch em back up!
        return y_nn

    def PEHEnn(self,concat_pred):
        p = self.split_pred(concat_pred)
        y_nn = self.find_ynn(p['phi']) #now its 3 plus because 
        cate_nn_err=tf.reduce_mean( tf.square( (1-2*self.data['t']) * (y_nn-self.data['y']) - (p['y1_pred']-p['y0_pred']) ) )
        return cate_nn_err

    def ATE(self,concat_pred):
        p = self.split_pred(concat_pred)
        return p['y1_pred']-p['y0_pred']

    def PEHE(self,concat_pred):
        #simulation only
        p = self.split_pred(concat_pred)
        cate_err=tf.reduce_mean( tf.square( ( (self.data['mu_1']-self.data['mu_0']) - (p['y1_pred']-p['y0_pred']) ) ) )
        return cate_err 

    def on_epoch_end(self, epoch, logs={}):
        concat_pred=self.model.predict(self.data['x'])
        #Calculate Empirical Metrics        
        ate_pred=tf.reduce_mean(self.ATE(concat_pred)); tf.summary.scalar('ate', data=ate_pred, step=epoch)
        pehe_nn=self.PEHEnn(concat_pred); tf.summary.scalar('cate_nn_err', data=tf.sqrt(pehe_nn), step=epoch)
        
        #Simulation Metrics
        ate_true=tf.reduce_mean(self.data['mu_1']-self.data['mu_0'])
        ate_err=tf.abs(ate_true-ate_pred); tf.summary.scalar('ate_err', data=ate_err, step=epoch)
        pehe =self.PEHE(concat_pred); tf.summary.scalar('cate_err', data=tf.sqrt(pehe), step=epoch)
        out_str=f' — ate_err: {ate_err:.4f}  — cate_err: {tf.sqrt(pehe):.4f} — cate_nn_err: {tf.sqrt(pehe_nn):.4f} '
        
        if self.verbose > 0: print(out_str)

