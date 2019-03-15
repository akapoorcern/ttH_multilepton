import numpy as np
import pandas
import os
import sklearn
import subprocess
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

class plotter(object):

    def __init__(self):
        self.separations_categories = []
        self.output_directory = ''
        pass

    def save_plots(self, dir='plots/', filename=''):
        self.check_dir(dir)
        filepath = os.path.join(dir,filename)
        self.fig.savefig(filepath)
        return self.fig

    def check_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def plot_training_progress_acc(self, histories, labels):
        self.fig, self.ax1 = plt.subplots(ncols=1, figsize=(10,10))
        for i in xrange(len(histories)):
            history1 = histories[i]
            label_name_train = '%s train' % (labels[i])
            label_name_test = '%s test' % (labels[i])
            #plt.plot(history1.history['acc'], label=label_name_train)
            #plt.plot(history1.history['val_acc'], label=label_name_test)
            plt.plot(history1.history['loss'], label=label_name_train)
            plt.plot(history1.history['val_loss'], label=label_name_test)

        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        acc_title = 'plots/DNN_accuracy_wrt_epoch.png'
        plt.tight_layout()
        return

    def correlation_matrix(self, data, **kwds):

        #iloc[<row selection>,<column selection>]
        self.data = data.iloc[:, :-4]
        self.labels = self.data.corr(**kwds).columns.values
        self.fig, self.ax1 = plt.subplots(ncols=1, figsize=(10,10))
        opts = {"annot" : True, "ax" : self.ax1, "vmin" : 0, "vmax" : 1*100, "annot_kws" : {"size":8}, "cmap" : plt.get_cmap("Blues",20), 'fmt' : '0.2f',}
        self.ax1.set_title("Correlations")
        sns.heatmap(self.data.corr(method='spearman')*100, **opts)
        for ax in (self.ax1,):
            # Shift tick location to bin centre
            ax.set_xticks(np.arange(len(self.labels))+0.5, minor=False)
            ax.set_yticks(np.arange(len(self.labels))+0.5, minor=False)
            ax.set_xticklabels(self.labels, minor=False, ha='right', rotation=45)
            ax.set_yticklabels(np.flipud(self.labels), minor=False, rotation=45)

        plt.tight_layout()

        return

    def conf_matrix(self, y_true, y_predicted,norm='index'):
        y_true = pandas.Series(y_true , name='truth')
        y_predicted = pandas.Series(y_predicted,name='prediction')
        self.matrix = pandas.crosstab(y_true,y_predicted,normalize=norm) #for classification accuracy
        #self.matrix = pandas.crosstab(y_true,y_predicted,normalize='columns') #for node purities
        print 'self.matrix.columns:'
        print self.matrix.columns
        self.labelsx = self.matrix.columns
        self.labelsy = self.matrix.index
        self.fig, self.ax1 = plt.subplots(ncols=1, figsize=(10,10))
        plt.rcParams.update({'font.size': 22})
        opts = {"annot" : True, "ax" : self.ax1, "vmin" : 0, "vmax" : 1, "annot_kws" : {"size":18}, "cmap" : plt.get_cmap("Reds",20), 'fmt' : '0.2f',}
        self.ax1.set_title("Confusion Matrix", fontsize=18)
        sns.set(font_scale=2.4)
        sns.heatmap(self.matrix, **opts)
        label_dict = {
            0 : 'ttH',
            1 : 'ttJ',
            2 : 'ttW',
            3 : 'ttZ'
        }
        for ax in (self.ax1,):
            #Shift tick location to bin centre
            ax.set_ylabel(y_true.name, fontsize = 18)
            ax.set_xlabel(y_predicted.name, fontsize = 18)
            ax.set_xticks(np.arange(len(self.labelsx))+0.5, minor=False)
            ax.set_yticks(np.arange(len(self.labelsy))+0.5, minor=False)
            print 'self.labelsx: ', self.labelsx
            new_xlabel = []
            new_ylabel = []
            for xlabel in self.labelsx:
                new_xlabel.append(label_dict.get(xlabel))
            print 'self.labelsy[::-1]: ', self.labelsy[::-1]
            for ylabel in self.labelsy[::-1]:
                new_ylabel.append(label_dict.get(ylabel))
            ax.set_xticklabels(new_xlabel, minor=False, ha='right', rotation=45, fontsize=18)
            ax.set_yticklabels(new_ylabel, minor=False, rotation=45, fontsize=18)
        plt.tight_layout()
        return


    def GetSeparation(self, hist_sig, hist_bckg):

        # compute "separation" defined as
        # <s2> = (1/2) Int_-oo..+oo { (S(x) - B(x))^2/(S(x) + B(x)) dx }
        separation = 0;
        # sanity check: signal and background histograms must have same number of bins and same limits
        if len(hist_sig) != len(hist_bckg):
            print 'Number of bins different for sig. and bckg'

        nBins = len(hist_sig)
        #dX = (hist_sig.GetXaxis().GetXmax() - hist_sig.GetXaxis().GetXmin()) / len(hist_sig)
        nS = np.sum(hist_sig)#*dX
        nB = np.sum(hist_bckg)#*dX

        if nS == 0:
            print 'WARNING: no signal weights'
        if nB == 0:
            print 'WARNING: no bckg weights'

        for i in xrange(1,nBins):
            sig_bin_norm = hist_sig[i]/nS
            bckg_bin_norm = hist_bckg[i]/nB
            # Separation:
            if(sig_bin_norm+bckg_bin_norm > 0):
                separation += 0.5 * ((sig_bin_norm - bckg_bin_norm) * (sig_bin_norm - bckg_bin_norm)) / (sig_bin_norm + bckg_bin_norm)
        #separation *= dX
        return separation


    def draw_category_overfitting_plot(self, y_scores_train, y_scores_test, plot_info):
        labels = plot_info[0]
        colours = plot_info[1]
        data_type = plot_info[2]
        plots_dir = plot_info[3]
        node_name = plot_info[4]
        plot_title = plot_info[5]
        name = filter(str.isalnum, str(data_type).split(".")[-1])
        self.fig, self.ax = plt.subplots(figsize=(8,6))
        self.ax.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
        self.ax.set_facecolor('white')

        #bin_edges_low_high = np.linspace(0.,1.,nbins+1)
        bin_edges_low_high = np.array([0.,0.0625,0.125,0.1875,0.25,0.3125,0.375,0.4375,0.5,0.5625,0.6125,0.6875,0.75,0.8125,0.875,0.9375,1.0])

        for index in xrange(len(y_scores_train)):
            y_train = y_scores_train[index]
            y_test = y_scores_test[index]
            label = labels[index]
            colour = colours[index]

            trainlabel = label + ' train'
            width = np.diff(bin_edges_low_high)
            histo_train_, bin_edges = np.histogram(y_train, bins=bin_edges_low_high)
            bincenters = 0.5*(bin_edges[1:]+bin_edges[:-1])
            dx_scale_train =(bin_edges[len(bin_edges)-1] - bin_edges[0]) / (len(bin_edges)-1)
            histo_train_ = histo_train_ / np.sum(histo_train_, dtype=np.float32) / dx_scale_train
            plt.bar(bincenters, histo_train_, width=width, color=colour, edgecolor=colour, alpha=0.5, label=trainlabel)

            if index == 0:
                histo_train_ttH = histo_train_ / np.sum(histo_train_, dtype=np.float32)
            if index == 1:
                histo_train_ttJ = histo_train_ / np.sum(histo_train_, dtype=np.float32)
            if index == 2:
                histo_train_ttW = histo_train_ / np.sum(histo_train_, dtype=np.float32)
            if index == 3:
                histo_train_ttZ = histo_train_ / np.sum(histo_train_, dtype=np.float32)

            testlabel = label + ' test'
            histo_test_, bin_edges = np.histogram(y_test, bins=bin_edges_low_high)
            dx_scale_test =(bin_edges[len(bin_edges)-1] - bin_edges[0]) / (len(bin_edges)-1)
            err = np.sqrt(histo_test_/np.sum(histo_test_, dtype=np.float32))
            histo_test_ = histo_test_ / np.sum(histo_test_, dtype=np.float32) / dx_scale_test
            bincenters = 0.5*(bin_edges[1:]+bin_edges[:-1])
            plt.errorbar(bincenters, histo_test_, yerr=err, fmt='o', c=colour, label=testlabel)

            if index == 0:
                histo_test_ttH = histo_test_ / np.sum(histo_test_, dtype=np.float32)
            if index == 1:
                histo_test_ttJ = histo_test_ / np.sum(histo_test_, dtype=np.float32)
            if index == 2:
                histo_test_ttW = histo_test_ / np.sum(histo_test_, dtype=np.float32)
            if index == 3:
                histo_test_ttZ = histo_test_ / np.sum(histo_test_, dtype=np.float32)

        if 'ttH' in node_name:
            print 'histo_train_ttH integral = ', histo_train_ttH.sum()
            train_ttHvttJSep = "{0:.5g}".format(self.GetSeparation(histo_train_ttH,histo_train_ttJ))
            train_ttHvttWSep = "{0:.5g}".format(self.GetSeparation(histo_train_ttH,histo_train_ttW))
            train_ttHvttZSep = "{0:.5g}".format(self.GetSeparation(histo_train_ttH,histo_train_ttZ))
            test_ttHvttJSep = "{0:.5g}".format(self.GetSeparation(histo_test_ttH,histo_test_ttJ))
            test_ttHvttWSep = "{0:.5g}".format(self.GetSeparation(histo_test_ttH,histo_test_ttW))
            test_ttHvttZSep = "{0:.5g}".format(self.GetSeparation(histo_test_ttH,histo_test_ttZ))

            ttH_v_ttJ_train_sep = 'ttH vs ttJ train Sep.: %s' % ( train_ttHvttJSep )
            self.ax.annotate(ttH_v_ttJ_train_sep,  xy=(0.7, 2.5), xytext=(0.7, 2.5), fontsize=9)
            ttH_v_ttW_train_sep = 'ttH vs ttW train Sep.: %s' % ( train_ttHvttWSep )
            self.ax.annotate(ttH_v_ttW_train_sep,  xy=(0.7, 2.25), xytext=(0.7, 2.25), fontsize=9)
            ttH_v_ttZ_train_sep = 'ttH vs ttW train Sep.: %s' % ( train_ttHvttZSep )
            self.ax.annotate(ttH_v_ttZ_train_sep,  xy=(0.7, 2.), xytext=(0.7, 2.), fontsize=9)
            ttH_v_ttJ_test_sep = 'ttH vs ttJ test Sep.: %s' % ( test_ttHvttJSep )
            self.ax.annotate(ttH_v_ttJ_test_sep,  xy=(0.7, 1.75), xytext=(0.7, 1.75), fontsize=9)
            ttH_v_ttW_test_sep = 'ttH vs ttW test Sep.: %s' % ( test_ttHvttWSep )
            self.ax.annotate(ttH_v_ttW_test_sep,  xy=(0.7, 1.5), xytext=(0.7, 1.5), fontsize=9)
            ttH_v_ttZ_test_sep = 'ttH vs ttZ test Sep.: %s' % ( test_ttHvttZSep )
            self.ax.annotate(ttH_v_ttZ_test_sep,  xy=(0.7, 1.25), xytext=(0.7, 1.25), fontsize=9)
            separations_forTable = r'''\textbackslash & %s & %s & %s''' % (test_ttHvttJSep, test_ttHvttWSep, test_ttHvttZSep)
        if 'ttJ' in node_name:
            train_ttJvttH = "{0:.5g}".format(self.GetSeparation(histo_train_ttJ,histo_train_ttH))
            train_ttJvttW = "{0:.5g}".format(self.GetSeparation(histo_train_ttJ,histo_train_ttW))
            train_ttJvttZ = "{0:.5g}".format(self.GetSeparation(histo_train_ttJ,histo_train_ttZ))
            test_ttJvttH = "{0:.5g}".format(self.GetSeparation(histo_test_ttJ,histo_test_ttH))
            test_ttJvttW = "{0:.5g}".format(self.GetSeparation(histo_test_ttJ,histo_test_ttW))
            test_ttJvttZ = "{0:.5g}".format(self.GetSeparation(histo_test_ttJ,histo_test_ttZ))

            ttJ_v_ttH_train_sep = 'ttJ vs ttH train Sep.: %s' % ( train_ttJvttH )
            self.ax.annotate(ttJ_v_ttH_train_sep,  xy=(0.7, 2.5), xytext=(0.7, 2.5), fontsize=9)
            ttJ_v_ttW_train_sep = 'ttJ vs ttW train Sep.: %s' % ( train_ttJvttW )
            self.ax.annotate(ttJ_v_ttW_train_sep,  xy=(0.7, 2.25), xytext=(0.7, 2.25), fontsize=9)
            ttJ_v_ttZ_train_sep = 'ttJ vs ttW train Sep.: %s' % ( train_ttJvttZ )
            self.ax.annotate(ttJ_v_ttZ_train_sep,  xy=(0.7, 2.), xytext=(0.7, 2.), fontsize=9)
            ttJ_v_ttH_test_sep = 'ttJ vs ttH test Sep.: %s' % ( test_ttJvttH )
            self.ax.annotate(ttJ_v_ttH_test_sep,  xy=(0.7, 1.75), xytext=(0.7, 1.75), fontsize=9)
            ttJ_v_ttW_test_sep = 'ttJ vs ttW test Sep.: %s' % ( test_ttJvttW )
            self.ax.annotate(ttJ_v_ttW_test_sep,  xy=(0.7, 1.5), xytext=(0.7, 1.5), fontsize=9)
            ttJ_v_ttZ_test_sep = 'ttJ vs ttZ test Sep.: %s' % ( test_ttJvttZ )
            self.ax.annotate(ttJ_v_ttZ_test_sep,  xy=(0.7, 1.25), xytext=(0.7, 1.25), fontsize=9)
            separations_forTable = r'''%s & \textbackslash & %s & %s''' % (test_ttJvttH,test_ttJvttW,test_ttJvttZ)
        if 'ttW' in node_name:
            train_ttWvttH = "{0:.5g}".format(self.GetSeparation(histo_train_ttW,histo_train_ttH))
            train_ttWvttJ = "{0:.5g}".format(self.GetSeparation(histo_train_ttW,histo_train_ttJ))
            train_ttWvttZ = "{0:.5g}".format(self.GetSeparation(histo_train_ttW,histo_train_ttZ))
            test_ttWvttH = "{0:.5g}".format(self.GetSeparation(histo_test_ttW,histo_test_ttH))
            test_ttWvttJ = "{0:.5g}".format(self.GetSeparation(histo_test_ttW,histo_test_ttJ))
            test_ttWvttZ = "{0:.5g}".format(self.GetSeparation(histo_test_ttW,histo_test_ttZ))

            ttW_v_ttH_train_sep = 'ttW vs ttH train Sep.: %s' % ( train_ttWvttH )
            self.ax.annotate(ttW_v_ttH_train_sep,  xy=(0.7, 2.5), xytext=(0.7, 2.5), fontsize=9)
            ttW_v_ttJ_train_sep = 'ttW vs ttJ train Sep.: %s' % ( train_ttWvttJ )
            self.ax.annotate(ttW_v_ttJ_train_sep,  xy=(0.7, 2.25), xytext=(0.7, 2.25), fontsize=9)
            ttW_v_ttZ_train_sep = 'ttW vs ttW train Sep.: %s' % ( train_ttWvttZ )
            self.ax.annotate(ttW_v_ttZ_train_sep,  xy=(0.7, 2.), xytext=(0.7, 2.), fontsize=9)
            ttW_v_ttH_test_sep = 'ttW vs ttH test Sep.: %s' % ( test_ttWvttH )
            self.ax.annotate(ttW_v_ttH_test_sep,  xy=(0.7, 1.75), xytext=(0.7, 1.75), fontsize=9)
            ttW_v_ttJ_test_sep = 'ttW vs ttJ test Sep.: %s' % ( test_ttWvttJ )
            self.ax.annotate(ttW_v_ttJ_test_sep,  xy=(0.7, 1.5), xytext=(0.7, 1.5), fontsize=9)
            ttW_v_ttZ_test_sep = 'ttW vs ttZ test Sep.: %s' % ( test_ttWvttZ )
            self.ax.annotate(ttW_v_ttZ_test_sep,  xy=(0.7, 1.25), xytext=(0.7, 1.25), fontsize=9)
            separations_forTable = r'''%s & %s & \textbackslash & %s''' % (test_ttWvttH,test_ttWvttJ,test_ttWvttZ)
        if 'ttZ' in node_name:
            train_ttZvttH = "{0:.5g}".format(self.GetSeparation(histo_train_ttZ,histo_train_ttH))
            train_ttZvttJ = "{0:.5g}".format(self.GetSeparation(histo_train_ttZ,histo_train_ttJ))
            train_ttZvttW = "{0:.5g}".format(self.GetSeparation(histo_train_ttZ,histo_train_ttW))
            test_ttZvttH = "{0:.5g}".format(self.GetSeparation(histo_test_ttZ,histo_test_ttH))
            test_ttZvttJ = "{0:.5g}".format(self.GetSeparation(histo_test_ttZ,histo_test_ttJ))
            test_ttZvttW = "{0:.5g}".format(self.GetSeparation(histo_test_ttZ,histo_test_ttW))

            ttZ_v_ttH_train_sep = 'ttZ vs ttH train Sep.: %s' % ( train_ttZvttH )
            self.ax.annotate(ttZ_v_ttH_train_sep,  xy=(0.7, 2.5), xytext=(0.7, 2.5), fontsize=9)
            ttZ_v_ttJ_train_sep = 'ttZ vs ttJ train Sep.: %s' % ( train_ttZvttJ )
            self.ax.annotate(ttZ_v_ttJ_train_sep,  xy=(0.7, 2.25), xytext=(0.7, 2.25), fontsize=9)
            ttZ_v_ttW_train_sep = 'ttZ vs ttW train Sep.: %s' % ( train_ttZvttW )
            self.ax.annotate(ttZ_v_ttW_train_sep,  xy=(0.7, 2.), xytext=(0.7, 2.), fontsize=9)
            ttZ_v_ttH_test_sep = 'ttZ vs ttH test Sep.: %s' % ( test_ttZvttH )
            self.ax.annotate(ttZ_v_ttH_test_sep,  xy=(0.7, 1.75), xytext=(0.7, 1.75), fontsize=9)
            ttZ_v_ttJ_test_sep = 'ttZ vs ttJ test Sep.: %s' % ( test_ttZvttJ )
            self.ax.annotate(ttZ_v_ttJ_test_sep,  xy=(0.7, 1.5), xytext=(0.7, 1.5), fontsize=9)
            ttZ_v_ttW_test_sep = 'ttZ vs ttW test Sep.: %s' % ( test_ttZvttW )
            self.ax.annotate(ttZ_v_ttW_test_sep,  xy=(0.7, 1.25), xytext=(0.7, 1.25), fontsize=9)
            separations_forTable = r'''%s & %s & %s & \textbackslash ''' % (test_ttZvttH,test_ttZvttJ,test_ttZvttW)

        title_ = '%s %s node' % (plot_title,node_name)
        plt.title(title_)
        label_name = 'DNN Output Score'
        plt.xlabel(label_name)
        plt.ylabel('(1/N)dN/dX')

        leg = plt.legend(loc='best', frameon=False, fancybox=False, fontsize=9)
        leg.get_frame().set_edgecolor('w')
        frame = leg.get_frame()
        frame.set_facecolor('White')


        overfitting_plot_file_name = 'overfitting_plot_%s_%s.png' % (node_name,plot_title)
        print 'Saving : %s%s' % (plots_dir, overfitting_plot_file_name)
        self.save_plots(dir=plots_dir, filename=overfitting_plot_file_name)

        return separations_forTable


    def separation_table(self , outputdir):
        content = r'''\documentclass{article}
\begin{document}
\begin{center}
\begin{table}
\begin{tabular}{| c | c | c | c | c |} \hline
Node \textbackslash Background & ttH & ttJ & ttW & ttZ \\ \hline
ttH & %s \\
ttJ & %s \\
ttW & %s \\
ttZ & %s \\ \hline
\end{tabular}
\caption{Separation power on each output node. The separation is given with respect to the `signal' process the node is trained to separate (one node per row) and the background processes for that node (one background per column).}
\end{table}
\end{center}
\end{document}
'''
        table_path = os.path.join(outputdir,'separation_table')
        table_tex = table_path+'.tex'
        print 'table_tex: ', table_tex
        with open(table_tex,'w') as f:
            f.write( content % (self.separations_categories[0], self.separations_categories[1], self.separations_categories[2], self.separations_categories[3] ) )
        return

    def overfitting(self, estimator, Y_train, Y_test, result_probs, result_probs_test, plots_dir, train_weights, test_weights, nbins=50):

        model = estimator
        data_type = type(model)


        #Arrays to store all ttH values
        y_scores_train_ttH_sample_ttHnode = []
        y_scores_train_ttJ_sample_ttHnode = []
        y_scores_train_ttW_sample_ttHnode = []
        y_scores_train_ttZ_sample_ttHnode = []

        # Arrays to store ttH categorised event values
        y_scores_train_ttH_sample_ttH_categorised = []
        y_scores_train_ttJ_sample_ttH_categorised = []
        y_scores_train_ttW_sample_ttH_categorised = []
        y_scores_train_ttZ_sample_ttH_categorised = []

        # Arrays to store all ttJ node values
        y_scores_train_ttH_sample_ttJnode = []
        y_scores_train_ttJ_sample_ttJnode = []
        y_scores_train_ttW_sample_ttJnode = []
        y_scores_train_ttZ_sample_ttJnode = []

        # Arrays to store ttJ categorised event values
        y_scores_train_ttH_sample_ttJ_categorised = []
        y_scores_train_ttJ_sample_ttJ_categorised = []
        y_scores_train_ttW_sample_ttJ_categorised = []
        y_scores_train_ttZ_sample_ttJ_categorised = []

        # Arrays to store all ttW node values
        y_scores_train_ttH_sample_ttWnode = []
        y_scores_train_ttJ_sample_ttWnode = []
        y_scores_train_ttW_sample_ttWnode = []
        y_scores_train_ttZ_sample_ttWnode = []

        # Arrays to store ttW categorised events
        y_scores_train_ttH_sample_ttW_categorised = []
        y_scores_train_ttJ_sample_ttW_categorised = []
        y_scores_train_ttW_sample_ttW_categorised = []
        y_scores_train_ttZ_sample_ttW_categorised = []

        # Arrays to store all ttZ node values
        y_scores_train_ttH_sample_ttZnode = []
        y_scores_train_ttJ_sample_ttZnode = []
        y_scores_train_ttW_sample_ttZnode = []
        y_scores_train_ttZ_sample_ttZnode = []

        # Arrays to store ttZ categorised events
        y_scores_train_ttH_sample_ttZ_categorised = []
        y_scores_train_ttJ_sample_ttZ_categorised = []
        y_scores_train_ttW_sample_ttZ_categorised = []
        y_scores_train_ttZ_sample_ttZ_categorised = []

        for i in xrange(len(result_probs)):
            train_event_weight = train_weights[i]
            if Y_train[i][0] == 1:
                y_scores_train_ttH_sample_ttHnode.append(result_probs[i][0])
                y_scores_train_ttH_sample_ttJnode.append(result_probs[i][1])
                y_scores_train_ttH_sample_ttWnode.append(result_probs[i][2])
                y_scores_train_ttH_sample_ttZnode.append(result_probs[i][3])
                # Get index of maximum argument.
                if np.argmax(result_probs[i]) == 0:
                    y_scores_train_ttH_sample_ttH_categorised.append(result_probs[i][0])
                if np.argmax(result_probs[i]) == 1:
                    y_scores_train_ttH_sample_ttJ_categorised.append(result_probs[i][1])
                if np.argmax(result_probs[i]) == 2:
                    y_scores_train_ttH_sample_ttW_categorised.append(result_probs[i][2])
                if np.argmax(result_probs[i]) == 3:
                    y_scores_train_ttH_sample_ttZ_categorised.append(result_probs[i][3])
            if Y_train[i][1] == 1:
                y_scores_train_ttJ_sample_ttHnode.append(result_probs[i][0])
                y_scores_train_ttJ_sample_ttJnode.append(result_probs[i][1])
                y_scores_train_ttJ_sample_ttWnode.append(result_probs[i][2])
                y_scores_train_ttJ_sample_ttZnode.append(result_probs[i][3])
                if np.argmax(result_probs[i]) == 0:
                    y_scores_train_ttJ_sample_ttH_categorised.append(result_probs[i][0])
                if np.argmax(result_probs[i]) == 1:
                    y_scores_train_ttJ_sample_ttJ_categorised.append(result_probs[i][1])
                if np.argmax(result_probs[i]) == 2:
                    y_scores_train_ttJ_sample_ttW_categorised.append(result_probs[i][2])
                if np.argmax(result_probs[i]) == 3:
                    y_scores_train_ttJ_sample_ttZ_categorised.append(result_probs[i][3])
            if Y_train[i][2] == 1:
                y_scores_train_ttW_sample_ttHnode.append(result_probs[i][0])
                y_scores_train_ttW_sample_ttJnode.append(result_probs[i][1])
                y_scores_train_ttW_sample_ttWnode.append(result_probs[i][2])
                y_scores_train_ttW_sample_ttZnode.append(result_probs[i][3])
                if np.argmax(result_probs[i]) == 0:
                    y_scores_train_ttW_sample_ttH_categorised.append(result_probs[i][0])
                if np.argmax(result_probs[i]) == 1:
                    y_scores_train_ttW_sample_ttJ_categorised.append(result_probs[i][1])
                if np.argmax(result_probs[i]) == 2:
                    y_scores_train_ttW_sample_ttW_categorised.append(result_probs[i][2])
                if np.argmax(result_probs[i]) == 3:
                    y_scores_train_ttW_sample_ttZ_categorised.append(result_probs[i][3])
            if Y_train[i][3] == 1:
                y_scores_train_ttZ_sample_ttHnode.append(result_probs[i][0])
                y_scores_train_ttZ_sample_ttJnode.append(result_probs[i][1])
                y_scores_train_ttZ_sample_ttWnode.append(result_probs[i][2])
                y_scores_train_ttZ_sample_ttZnode.append(result_probs[i][3])
                if np.argmax(result_probs[i]) == 0:
                    y_scores_train_ttZ_sample_ttH_categorised.append(result_probs[i][0])
                if np.argmax(result_probs[i]) == 1:
                    y_scores_train_ttZ_sample_ttJ_categorised.append(result_probs[i][1])
                if np.argmax(result_probs[i]) == 2:
                    y_scores_train_ttZ_sample_ttW_categorised.append(result_probs[i][2])
                if np.argmax(result_probs[i]) == 3:
                    y_scores_train_ttZ_sample_ttZ_categorised.append(result_probs[i][3])

        #Arrays to store all ttH values
        y_scores_test_ttH_sample_ttHnode = []
        y_scores_test_ttJ_sample_ttHnode = []
        y_scores_test_ttW_sample_ttHnode = []
        y_scores_test_ttZ_sample_ttHnode = []

        # Arrays to store ttH categorised event values
        y_scores_test_ttH_sample_ttH_categorised = []
        y_scores_test_ttJ_sample_ttH_categorised = []
        y_scores_test_ttW_sample_ttH_categorised = []
        y_scores_test_ttZ_sample_ttH_categorised = []

        # Arrays to store all ttJ node values
        y_scores_test_ttH_sample_ttJnode = []
        y_scores_test_ttJ_sample_ttJnode = []
        y_scores_test_ttW_sample_ttJnode = []
        y_scores_test_ttZ_sample_ttJnode = []

        # Arrays to store ttJ categorised event values
        y_scores_test_ttH_sample_ttJ_categorised = []
        y_scores_test_ttJ_sample_ttJ_categorised = []
        y_scores_test_ttW_sample_ttJ_categorised = []
        y_scores_test_ttZ_sample_ttJ_categorised = []

        # Arrays to store all ttW node values
        y_scores_test_ttH_sample_ttWnode = []
        y_scores_test_ttJ_sample_ttWnode = []
        y_scores_test_ttW_sample_ttWnode = []
        y_scores_test_ttZ_sample_ttWnode = []

        # Arrays to store ttW categorised events
        y_scores_test_ttH_sample_ttW_categorised = []
        y_scores_test_ttJ_sample_ttW_categorised = []
        y_scores_test_ttW_sample_ttW_categorised = []
        y_scores_test_ttZ_sample_ttW_categorised = []

        # Arrays to store all ttZ node values
        y_scores_test_ttH_sample_ttZnode = []
        y_scores_test_ttJ_sample_ttZnode = []
        y_scores_test_ttW_sample_ttZnode = []
        y_scores_test_ttZ_sample_ttZnode = []

        # Arrays to store ttZ categorised events
        y_scores_test_ttH_sample_ttZ_categorised = []
        y_scores_test_ttJ_sample_ttZ_categorised = []
        y_scores_test_ttW_sample_ttZ_categorised = []
        y_scores_test_ttZ_sample_ttZ_categorised = []
        for i in xrange(len(result_probs_test)):
            test_event_weight = test_weights[i]
            if Y_test[i][0] == 1:
                y_scores_test_ttH_sample_ttHnode.append(result_probs_test[i][0])
                y_scores_test_ttH_sample_ttJnode.append(result_probs_test[i][1])
                y_scores_test_ttH_sample_ttWnode.append(result_probs_test[i][2])
                y_scores_test_ttH_sample_ttZnode.append(result_probs_test[i][3])
                if np.argmax(result_probs_test[i]) == 0:
                    y_scores_test_ttH_sample_ttH_categorised.append(result_probs_test[i][0])
                if np.argmax(result_probs_test[i]) == 1:
                    y_scores_test_ttH_sample_ttJ_categorised.append(result_probs_test[i][1])
                if np.argmax(result_probs_test[i]) == 2:
                    y_scores_test_ttH_sample_ttW_categorised.append(result_probs_test[i][2])
                if np.argmax(result_probs_test[i]) == 3:
                    y_scores_test_ttH_sample_ttZ_categorised.append(result_probs_test[i][3])
            if Y_test[i][1] == 1:
                y_scores_test_ttJ_sample_ttHnode.append(result_probs_test[i][0])
                y_scores_test_ttJ_sample_ttJnode.append(result_probs_test[i][1])
                y_scores_test_ttJ_sample_ttWnode.append(result_probs_test[i][2])
                y_scores_test_ttJ_sample_ttZnode.append(result_probs_test[i][3])
                if np.argmax(result_probs_test[i]) == 0:
                    y_scores_test_ttJ_sample_ttH_categorised.append(result_probs_test[i][0])
                if np.argmax(result_probs_test[i]) == 1:
                    y_scores_test_ttJ_sample_ttJ_categorised.append(result_probs_test[i][1])
                if np.argmax(result_probs_test[i]) == 2:
                    y_scores_test_ttJ_sample_ttW_categorised.append(result_probs_test[i][2])
                if np.argmax(result_probs_test[i]) == 3:
                    y_scores_test_ttJ_sample_ttZ_categorised.append(result_probs_test[i][3])
            if Y_test[i][2] == 1:
                y_scores_test_ttW_sample_ttHnode.append(result_probs_test[i][0])
                y_scores_test_ttW_sample_ttJnode.append(result_probs_test[i][1])
                y_scores_test_ttW_sample_ttWnode.append(result_probs_test[i][2])
                y_scores_test_ttW_sample_ttZnode.append(result_probs_test[i][3])
                if np.argmax(result_probs_test[i]) == 0:
                    y_scores_test_ttW_sample_ttH_categorised.append(result_probs_test[i][0])
                if np.argmax(result_probs_test[i]) == 1:
                    y_scores_test_ttW_sample_ttJ_categorised.append(result_probs_test[i][1])
                if np.argmax(result_probs_test[i]) == 2:
                    y_scores_test_ttW_sample_ttW_categorised.append(result_probs_test[i][2])
                if np.argmax(result_probs_test[i]) == 3:
                    y_scores_test_ttW_sample_ttZ_categorised.append(result_probs_test[i][3])
            if Y_test[i][3] == 1:
                y_scores_test_ttZ_sample_ttHnode.append(result_probs_test[i][0])
                y_scores_test_ttZ_sample_ttJnode.append(result_probs_test[i][1])
                y_scores_test_ttZ_sample_ttWnode.append(result_probs_test[i][2])
                y_scores_test_ttZ_sample_ttZnode.append(result_probs_test[i][3])
                if np.argmax(result_probs_test[i]) == 0:
                    y_scores_test_ttZ_sample_ttH_categorised.append(result_probs_test[i][0])
                if np.argmax(result_probs_test[i]) == 1:
                    y_scores_test_ttZ_sample_ttJ_categorised.append(result_probs_test[i][1])
                if np.argmax(result_probs_test[i]) == 2:
                    y_scores_test_ttZ_sample_ttW_categorised.append(result_probs_test[i][2])
                if np.argmax(result_probs_test[i]) == 3:
                    y_scores_test_ttZ_sample_ttZ_categorised.append(result_probs_test[i][3])

        w, h = 4, 4
        y_scores_train = [[0 for x in range(w)] for y in range(h)]
        y_scores_test = [[0 for x in range(w)] for y in range(h)]
        y_scores_train[0] = [y_scores_train_ttH_sample_ttH_categorised, y_scores_train_ttJ_sample_ttH_categorised, y_scores_train_ttW_sample_ttH_categorised, y_scores_train_ttZ_sample_ttH_categorised]
        y_scores_train[1] = [y_scores_train_ttH_sample_ttJ_categorised, y_scores_train_ttJ_sample_ttJ_categorised, y_scores_train_ttW_sample_ttJ_categorised, y_scores_train_ttZ_sample_ttJ_categorised]
        y_scores_train[2] = [y_scores_train_ttH_sample_ttW_categorised, y_scores_train_ttJ_sample_ttW_categorised, y_scores_train_ttW_sample_ttW_categorised, y_scores_train_ttZ_sample_ttW_categorised]
        y_scores_train[3] = [y_scores_train_ttH_sample_ttZ_categorised, y_scores_train_ttJ_sample_ttZ_categorised, y_scores_train_ttW_sample_ttZ_categorised, y_scores_train_ttZ_sample_ttZ_categorised]
        y_scores_test[0] = [y_scores_test_ttH_sample_ttH_categorised, y_scores_test_ttJ_sample_ttH_categorised, y_scores_test_ttW_sample_ttH_categorised, y_scores_test_ttZ_sample_ttH_categorised]
        y_scores_test[1] = [y_scores_test_ttH_sample_ttJ_categorised, y_scores_test_ttJ_sample_ttJ_categorised, y_scores_test_ttW_sample_ttJ_categorised, y_scores_test_ttZ_sample_ttJ_categorised]
        y_scores_test[2] = [y_scores_test_ttH_sample_ttW_categorised, y_scores_test_ttJ_sample_ttW_categorised, y_scores_test_ttW_sample_ttW_categorised, y_scores_test_ttZ_sample_ttW_categorised]
        y_scores_test[3] = [y_scores_test_ttH_sample_ttZ_categorised, y_scores_test_ttJ_sample_ttZ_categorised, y_scores_test_ttW_sample_ttZ_categorised, y_scores_test_ttZ_sample_ttZ_categorised]

        y_scores_train_nonCat = [[0 for x in range(w)] for y in range(h)]
        y_scores_test_nonCat = [[0 for x in range(w)] for y in range(h)]
        y_scores_train_nonCat[0] = [y_scores_train_ttH_sample_ttHnode, y_scores_train_ttJ_sample_ttHnode, y_scores_train_ttW_sample_ttHnode, y_scores_train_ttZ_sample_ttHnode]
        y_scores_train_nonCat[1] = [y_scores_train_ttH_sample_ttJnode, y_scores_train_ttJ_sample_ttJnode, y_scores_train_ttW_sample_ttJnode, y_scores_train_ttZ_sample_ttJnode]
        y_scores_train_nonCat[2] = [y_scores_train_ttH_sample_ttWnode, y_scores_train_ttJ_sample_ttWnode, y_scores_train_ttW_sample_ttWnode, y_scores_train_ttZ_sample_ttWnode]
        y_scores_train_nonCat[3] = [y_scores_train_ttH_sample_ttZnode, y_scores_train_ttJ_sample_ttZnode, y_scores_train_ttW_sample_ttZnode, y_scores_train_ttZ_sample_ttZnode]
        y_scores_test_nonCat[0] = [y_scores_test_ttH_sample_ttHnode, y_scores_test_ttJ_sample_ttHnode, y_scores_test_ttW_sample_ttHnode, y_scores_test_ttZ_sample_ttHnode]
        y_scores_test_nonCat[1] = [y_scores_test_ttH_sample_ttJnode, y_scores_test_ttJ_sample_ttJnode, y_scores_test_ttW_sample_ttJnode, y_scores_test_ttZ_sample_ttJnode]
        y_scores_test_nonCat[2] = [y_scores_test_ttH_sample_ttWnode, y_scores_test_ttJ_sample_ttWnode, y_scores_test_ttW_sample_ttWnode, y_scores_test_ttZ_sample_ttWnode]
        y_scores_test_nonCat[3] = [y_scores_test_ttH_sample_ttZnode, y_scores_test_ttJ_sample_ttZnode, y_scores_test_ttW_sample_ttZnode, y_scores_test_ttZ_sample_ttZnode]

        node_name = ['ttH','ttJ','ttW','ttZ']
        counter =0
        for y_scorestrain,y_scorestest in zip(y_scores_train,y_scores_test):
            colours = ['r','steelblue','g','Fuchsia']
            node_title = node_name[counter]
            plot_title = 'Categorised'
            plot_info = [node_name,colours,data_type,plots_dir,node_title,plot_title]
            self.separations_categories.append(self.draw_category_overfitting_plot(y_scorestrain,y_scorestest,plot_info))
            counter = counter +1

        counter =0
        separations_all = []
        for y_scores_train_nonCat,y_scores_test_nonCat in zip(y_scores_train_nonCat,y_scores_test_nonCat):
            colours = ['r','steelblue','g','Fuchsia']
            node_title = node_name[counter]
            plot_title = 'Non-Categorised'
            plot_info = [node_name,colours,data_type,plots_dir,node_title,plot_title]
            separations_all.append(self.draw_category_overfitting_plot(y_scores_train_nonCat,y_scores_test_nonCat,plot_info))
            counter = counter +1

        return
