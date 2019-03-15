import ROOT, os, optparse
from ROOT import TMVA, TFile, TString, TLegend, THStack, TLatex, TH1D
from array import array
from subprocess import call
from os.path import isfile
from collections import OrderedDict

class control_plotter(object):

    def __init__(self):
        self.output_directory = ''

    def save_plots(self, filename, dir='plots/'):
        self.check_dir(dir)
        filepath = os.path.join(dir,filename)
        self.fig.savefig(filepath)
        return self.fig

    def check_dir(self, dir):
        if not os.path.exists(dir):
            print 'Creating directory ',dir
            os.makedirs(dir)

    def getEOSlslist(self, directory, mask='', prepend='root://eosuser.cern.ch/'):
        from subprocess import Popen, PIPE

        eos_cmd = '/afs/cern.ch/project/eos/installation/0.3.15/bin/eos.select'
        eos_dir = '/eos/user/%s ' % (directory)
        data = Popen([eos_cmd, prepend, ' ls ', eos_dir], stdout=PIPE)
        out,err = data.communicate()

        full_list = []

        ## if input file was single root file:
        if directory.endswith('.root'):
            if len(out.split('\n')[0]) > 0:
                return [os.path.join(prepend,eos_dir).replace(" ","")]

        ## instead of only the file name append the string to open the file in ROOT
        for line in out.split('\n'):
            if len(line.split()) == 0: continue
            full_list.append(os.path.join(prepend,eos_dir,line).replace(" ",""))

        ## strip the list of files if required
        if mask != '':
            stripped_list = [x for x in full_list if mask in x]
            return stripped_list

        ## return
        return full_list

    def open_files(self, input_files_names):
        input_files_list = []
        print 'input_files_names: ', input_files_names
        for index in xrange(0,len(input_files_names)):
            input_files_list.append(TFile.Open(input_files_names[index]))
        return input_files_list

    def load_histos(self, input_files_, branches_, treename_):
        input_hist_dict = {}
        for file_ in input_files_:
            for branch_ in branches_:
                draw_command = '%s>>temph' % (branch_)
                file_.Get(treename_).Draw(draw_command)
                htemp = ROOT.gDirectory.Get("temph")
                htemp.SetName(branch_)
                input_hist_dict[file_.GetName()] = htemp
        return input_hist_dict

    def make_hist(self, input_hist_dict_):
        hist_count = 0
        c1 = ROOT.TCanvas('c1',',1000,1000')
        p1 = ROOT.TPad('p1','p1',0.0,0.2,1.0,1.0)
        p1.Draw()
        p1.SetRightMargin(0.1)
        p1.SetLeftMargin(0.1)
        p1.SetBottomMargin(0.1)
        p1.SetTopMargin(0.1)
        p1.SetGridx(True)
        p1.SetGridy(True)
        p1.cd()
        ROOT.gStyle.SetOptStat(0)
        ROOT.gStyle.SetOptTitle(0)
        for rootfile_key, hist in input_hist_dict_.iteritems():
            print 'hist.GetName()', hist.GetName()
            if hist_count == 0:
                hist.Draw()
            else:
                hist.Draw('same')
            hist_count = 1
            output_fullpath = 'invar_control_plots/' + hist.GetName()
        c1.Print(output_fullpath,'pdf')
        return

    def sum_hists(self, hists_):
        nbins = hists_[0].GetNbinsX()
        minX = hists_[0].GetMinimum()
        maxX = hists_[0].GetMaximum()
        combined_hist = ROOT.TH1F('summed_hist','summed_hist',nbins,minX,maxX)
        for hist_ in hists_:
            combined_hist.Add(hist_)
        return combined_hist
