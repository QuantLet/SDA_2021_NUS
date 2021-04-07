from scipy import stats
from scipy.stats import norm
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import copy
import pandas as pd
import datetime

class DrawPicture(object):
    def __init__(self, factor, direction, factor_data, fwrtn, date, mkt_index, Portfolio_return, coverage, Turnover, save_path_factor,Type):      
        self.factor = factor
        self.factor_data=factor_data
        if direction == "Descending":
            self.factor_data=-1*factor_data  
        self.date =date
        self.mkt_index = mkt_index
        self.Portfolio_return = Portfolio_return
        self.coverage = coverage
        self.Turnover = Turnover
        self.save_path = save_path_factor
        self.fwrtn = fwrtn
        self.Type = Type
        
    def drawDistrubution(self):
        dat = self.factor_data.iloc[-1]
        dat = dat.dropna()
        mean = np.mean(dat)
        std = np.std(dat)
        Y = norm(loc=mean,scale=std)
        step = np.max(dat) - np.min(dat)
        step = step / 100
        t = np.arange(np.min(dat)-10*step, np.max(dat)+10*step, step)
        
        plt.hist(dat,bins=40,normed=1, facecolor='blue', alpha=0.5)  
        plt.plot(t,Y.pdf(t),'r')
    
        plt.xlabel('Factor')
        plt.title(self.factor+ ' Distribution ')
        plt.savefig(os.path.join(self.save_path, 'Factor Distribution'))
        plt.show()
        
    def drawCoverage(self, win):#win = 48
        coverage = self.coverage
        date1 = pd.to_datetime(self.date,format = '%Y%m%d')
        plt.plot(date1,coverage)
        coverage2 = pd.Series(coverage).rolling(window=win,center=False).mean()
        plt.plot(date1, coverage2,'r')
        plt.ylabel('# of stock')
        plt.title('Coverage')
        plt.ylim(0,np.max(coverage)+500)
        plt.savefig(os.path.join(self.save_path, 'Coverage'))
        plt.show() 
        
    def drawTurnover(self, win):  ###win = 48
        """Calculates serial correlation and factor turnover from data of two adjacent months.
        """    
        fig, ax = plt.subplots()
        date1 = self.Turnover.index
        plt.plot(date1,self.Turnover)
        turnover2 = self.Turnover.rolling(window = win,center = False).mean()
        plt.plot(date1, turnover2,'r')
        plt.ylim(0, np.max(self.Turnover) + 5)
        
        plt.ylabel('Turnover / %')
        plt.title('Turnover')
        plt.savefig(os.path.join(self.save_path, 'Turnover'))
        plt.show()
        
    def drawQuantileRtn(self):#drawQuantileRtn
        #Calculates and saves the annualized volatility of the portfolio.
        data = copy.deepcopy(self.Portfolio_return)    
        #Calculates the annualized return of the portfolio, and saves it to the field.
        for i in range(len(self.Portfolio_return.columns)):
            net_rtn = self.Portfolio_return.iloc[:,i]
            data.ix[:,i]= (net_rtn+1).cumprod()
        #data.index = pd.to_datetime(self.Portfolio_return.index, format = '%Y%m%d')
        data.iloc[:,:-1].plot()
        plt.legend(loc=2,ncol=2,fancybox=True,shadow=True)
        leg = plt.gca().get_legend()
        ltext = leg.get_texts()
        plt.setp(ltext, fontsize='small')    
        plt.ylabel('net value')
        plt.title('Quantile Net Value')    
        plt.savefig(os.path.join(self.save_path, 'Quantile Net Value'))
        plt.show()
        data.to_csv(os.path.join(self.save_path, 'Quantile Net Value.csv'))
        self.data = data.iloc[:,:-1]
        return data
    
    def drawWealthCurve(self,win,year):
        lsnet = self.Portfolio_return.ix[:,-1]
        lsdata = (lsnet+1).cumprod() 
        self.lsdata = lsdata
        self.lsnet = lsnet
        drw = self.drawmaxDrawdown()

        # date1 = pd.to_datetime(self.date,format = '%Y%m%d')
        date1 = self.date
#        print (lsdata)
#        print (date1)
        plt.plot(date1,lsdata)

        #IR, MEAN, STDY
        plt.ylabel('net value')
        plt.title('Wealth Curve')
        length = lsdata.max() - lsdata.min()
        
        text1 = "IR: "+ str(round(self.IR_all[-1],2)) 
        text2 = "rtn: " + str(round((100*(lsdata.iloc[-1])**(win/len(lsdata))-100),2)) +" %"
        text3 = "std: " + str(round((np.std(100*lsnet)*math.sqrt(win)),2)) +" %"
        text4 = "turnover: " + str(round(self.Turnover.mean())) + " %"
        text5 = "drawdown: " + str(round(drw,2)) + " %"
        
        plt.text(datetime.datetime(year, 4, 4, 0, 0), np.max(lsdata), text1)  
        plt.text(datetime.datetime(year, 4, 4, 0, 0), np.max(lsdata)-0.08*length, text2)  
        plt.text(datetime.datetime(year, 4, 4, 0, 0), np.max(lsdata)-0.16*length, text3)
        plt.text(datetime.datetime(year, 4, 4, 0, 0), np.max(lsdata)-0.24*length, text4)
        plt.text(datetime.datetime(year, 4, 4, 0, 0), np.max(lsdata)-0.32*length, text5)

        plt.savefig(os.path.join(self.save_path, 'Wealth Curve'))
        plt.show()
        return lsdata
    
    def drawLogWealthCurve(self):
        date1 = pd.to_datetime(self.date,format = '%Y%m%d')
        plt.plot(date1,self.lsdata)
        plt.yscale('log')
        #IR, MEAN, STDY
        plt.ylabel('net value')
        plt.title('Wealth Curve(log)')
        plt.savefig(os.path.join(self.save_path, 'Wealth Curve(log)'))
        plt.show()

    def drawmaxDrawdown(self):
        drawdown = []
        for i in range(len(self.date)):
            di = self.lsdata.iloc[i]
            dwi = (di/np.max(self.lsdata[:(i+1)]))-1
            drawdown.append(dwi)
        drawdown = pd.Series(drawdown,index = self.date)*100
        date1 = pd.to_datetime(self.date,format = '%Y%m%d')
        plt.plot(date1,drawdown)
        #IR, MEAN, STDY
        plt.ylabel('drawdown(%)')
        plt.title('Drawdown')
        plt.savefig(os.path.join(self.save_path, 'Drawdown'))
        plt.show()
        return np.min(drawdown)

    def calAnualReturn(self, labels, win):
        rtn_all = []
        #Calculates the annualized return of the portfolio, and saves it to the field.
        for i in range(len(self.data.columns)):
            rtn = self.data.ix[:,i]
            rtn_all.append(100*(rtn[-1])**(win/len(rtn))-100)
            #print rtn_all[-1]
        ls = self.lsdata.dropna()
        rtn_all.append(100*(ls.iloc[-1])**(win/len(ls))-100)
        
        plt.ylim(np.min(rtn_all)-10, np.max(rtn_all)+10)
        # plt.xticks(range(len(labels)),labels,rotation=90)
        plt.xticks(range(len(labels)),labels)
        plt.ylabel('Annualized Return(%)')
        plt.title('Annualized Return')
        plt.bar(range(len(labels)), rtn_all,align="center")
        plt.savefig(os.path.join(self.save_path, 'Annualized Return'))
        plt.show()
        return rtn_all
    
    def calAnualVol(self,labels,win):
        #Calculates and saves the annualized volatility of the portfolio.
        std_all = []
        #Calculates the annualized return of the portfolio, and saves it to the field.
        for i in range(len(self.Portfolio_return.columns)):
            rtn = self.Portfolio_return.ix[:,i]*100
            std_all.append((np.std(rtn))*math.sqrt(win))
        plt.bar(range(len(labels)), std_all, align="center")
        # plt.xticks(range(len(labels)),labels,rotation=90)
        plt.xticks(range(len(labels)),labels)
        plt.ylabel('Annualized Volatility(%)')
        plt.title('Annualized Volatility(%)')
        plt.savefig(os.path.join(self.save_path, 'Annualized Volatility'))
        plt.show()
        return std_all
    def drawSampleRtn(self, win, year):
        #Calculates and saves the annualized volatility of the portfolio.
        time_series1 = self.Portfolio_return.ix[:,-1]*100
        time_series2 = time_series1.rolling(window=win,center=False).mean()
        date1 = pd.to_datetime(time_series1.index,format = '%Y%m%d')
        plt.bar(date1,time_series1,edgecolor = 'lightskyblue')
        plt.plot(date1,time_series2,'r')
        plt.ylim(np.min(time_series1)-2,np.max(time_series1)+4)
        
        avg = np.mean(time_series1)
        std = np.std(time_series1)
        min_net_rtn = np.min(time_series1)
        res = avg/std
        annual_avg = (1 + avg / 100) ** win * 100 - 100
        annual_std = std * np.sqrt(win)
        annual_res = annual_avg/ annual_std
        
        text1='avg: '+ str(round(avg,2)) +" %"
        text2 = 'std: '+str(round(std,2)) +" %"
        text3 = 'min: ' + str(round(min_net_rtn,2)) +" %"
        text4 = 'avg/std: ' +str(round(res,2)) +" %"
        text5 = 'annual avg/std: ' +str(round(annual_res,2)) +" %"
        
        length = np.max(time_series1)-np.min(time_series1)+6
        
        plt.text(datetime.datetime(year, 4, 4, 0, 0), np.max(time_series1)+4-length*0.1, text1)  
        plt.text(datetime.datetime(year, 4, 4, 0, 0), np.max(time_series1)+4-length*0.16, text2)  
        plt.text(datetime.datetime(year, 4, 4, 0, 0), np.max(time_series1)+4-length*0.22, text3)  
        plt.text(datetime.datetime(year, 4, 4, 0, 0), np.max(time_series1)+4-length*0.28, text4)  
        plt.text(datetime.datetime(year, 4, 4, 0, 0), np.max(time_series1)+4-length*0.34, text5)  

        plt.ylabel('Time Series Spread(%)')
        plt.title('Time Seires Spread(%)')
        plt.savefig(os.path.join(self.save_path, 'Time Seires Spread(%)'))
        plt.show()
    #serial correlation
    def calSerial(self, win,year):  ###
        """Calculates serial correlation and factor turnover from data of two adjacent months.
        """
        serial_cor=[]
        date1 = pd.to_datetime(self.date,format = '%Y%m%d')

        for i in range(1,len(date1)):
            lastweek = self.date[i-1]
            thisweek = self.date[i]
            lastWeekData =self.factor_data.ix[lastweek]
            thisWeekData =self.factor_data.ix[thisweek]
            stocks = list(set(lastWeekData.dropna().index) & set(thisWeekData.dropna().index))
            last = lastWeekData.ix[stocks]
            this = thisWeekData.ix[stocks]
            mu1, mu2, sigma1, sigma2 = np.nanmean(last), np.nanmean(this), np.nanstd(last), np.nanstd(this)
            if (sigma1 * sigma2==0):
                serial = 100
            else:
                serial = np.nanmean((last - mu1) * (this - mu2))/float(sigma1 * sigma2) * 100
            # print (serial)
            serial_cor.append(serial)
            
        serial_cor = pd.Series(serial_cor)
        serial_cor.index = date1[1:]
        serial_cor.to_csv(os.path.join(self.save_path, "Serial Correlation.csv"))

        serial_cor2=serial_cor.rolling(window=win,center=False).mean()
        plt.plot(date1[1:],serial_cor)
        plt.plot(date1[1:],serial_cor2,'r')
        plt.ylim(np.min(serial_cor)-10,np.max(serial_cor)+10)
                
        avg = np.mean(serial_cor)
        std = np.std(serial_cor)
        min_serial= np.min(serial_cor)
        res = avg/std
        
        text1='avg: '+ str(round(avg,2)) +" %"
        text2 = 'std: '+str(round(std,2)) +" %"
        text3 = 'min: ' + str(round(min_serial,2)) +" %" 
        text4 = 'avg/std: ' +str(round(res,2)) +" %"
        
        length = np.max(serial_cor) - np.min(serial_cor) + 20
                
        plt.text(datetime.datetime(year, 4, 4, 0, 0), np.max(serial_cor)+10-length*0.1, text1)  
        plt.text(datetime.datetime(year, 4, 4, 0, 0), np.max(serial_cor)+10-length*0.16, text2)  
        plt.text(datetime.datetime(year, 4, 4, 0, 0), np.max(serial_cor)+10-length*0.22, text3)  
        plt.text(datetime.datetime(year, 4, 4, 0, 0), np.max(serial_cor)+10-length*0.28, text4)  
    
        plt.ylabel('Serial corr(%)')
        plt.title('Serial Correlation(%)')

        plt.savefig(os.path.join(self.save_path, 'Serial Correlation'))
        plt.show()
        return serial_cor
    
    def calIR(self,mkt_index,win):
        IR_all = []   
        #Calculates the annualized return of the portfolio, and saves it to the field.
        for i in range(len(self.Portfolio_return.columns)-1):
            rtn = self.Portfolio_return.ix[:,i]
            IR_all.append(((np.nanmean(rtn-mkt_index))/np.nanstd(rtn-mkt_index))*math.sqrt(win))
        
        ls_rtn = self.Portfolio_return.ix[:,-1]
        IR_all.append((np.nanmean(ls_rtn)/np.nanstd(ls_rtn))*math.sqrt(win))
        plt.bar(range(len(self.Portfolio_return.columns)), IR_all, align="center")
        plt.ylabel('IR')
        plt.xticks(range(len(self.Portfolio_return.columns)), self.Portfolio_return.columns)
        plt.title('IR')
        plt.savefig(os.path.join(self.save_path, 'IR'))
        plt.show()
        self.IR_all = IR_all
        return IR_all

    def calSortinoRatio(self,mkt_index, win):
        SR_all = []
        for i in range(len(self.Portfolio_return.columns)-1):
            rtn = self.Portfolio_return.ix[:,i]
            DR=[]
            for day in rtn.index:
                DR.append(min(0,(rtn-mkt_index)[day]))
            SR_all.append(math.sqrt(win)*(np.mean(rtn-mkt_index))/np.std(DR)) 
        ls_rtn = self.Portfolio_return.ix[:,-1]
        DR= ls_rtn.copy()
        for day in ls_rtn.index:
            DR[day] = (min(0,ls_rtn[day]))
        SR_all.append(math.sqrt(win)*(np.mean(ls_rtn)/np.std(DR)))
        plt.bar(range(len(self.Portfolio_return.columns)), SR_all, align="center")
        plt.ylabel('Sortino Ratio')
        plt.xticks(range(len(self.Portfolio_return.columns)), self.Portfolio_return.columns)
        plt.title('Sortino Ratio')
        plt.savefig(os.path.join(self.save_path, 'Sortino Ratio'))
        plt.show()
        return SR_all
  
    def calSpearman(self,win,year):
        Spearman_cor=[]
#        date1 = pd.to_datetime(self.date,format = '%Y%m%d')

        for i in range(len(self.date)):#len(date1)):
            thisweek = self.date[i]
            thisWeekData =self.factor_data.ix[thisweek]
            rtndata = self.fwrtn.ix[thisweek]
            
            stocks = list(set(rtndata.dropna().index) & set(thisWeekData.dropna().index))
            #print (stocks)
            facs = thisWeekData.ix[stocks]
            fmrts = rtndata.ix[stocks]
            
            cor = stats.spearmanr(facs, fmrts).correlation * 100
            #print (cor)
            Spearman_cor.append(cor)
         
        Spearman_cor = pd.Series(Spearman_cor)
        Spearman_cor.index = self.date
        avg_IC = np.mean(Spearman_cor)
        std_IC = np.std(Spearman_cor)
        min_IC = np.min(Spearman_cor)
        max_IC = np.max(Spearman_cor)
        res = avg_IC/std_IC
        
        text1='avg: '+ str(round(avg_IC,2)) + " %"
        text2 = 'std: '+str(round(std_IC,2)) + " %"
        text3 = 'min: ' + str(round(min_IC,2)) + " %"
        text4 = 'max: ' + str(round(max_IC,2)) + " %"
        text5 = 'avg/std: ' +str(round(res,2)) 
                    
        
        Spearman_cor.to_csv(os.path.join(self.save_path, "Spearman IC.csv"))
        Spearman_cor2=Spearman_cor.rolling(window= win,center=False).mean()
        plt.bar(self.date,Spearman_cor,edgecolor = 'lightskyblue')
        plt.plot(self.date,Spearman_cor2,'r')
        plt.ylim(np.min(Spearman_cor)-10,np.max(Spearman_cor)+10)
        length = np.max(Spearman_cor)-np.min(Spearman_cor)+20
    
        plt.text(datetime.datetime(year, 4, 4, 0, 0), np.max(Spearman_cor)+10-length*0.1, text1)  
        plt.text(datetime.datetime(year, 4, 4, 0, 0), np.max(Spearman_cor)+10-length*0.16, text2)  
        plt.text(datetime.datetime(year, 4, 4, 0, 0), np.max(Spearman_cor)+10-length*0.22, text3)  
        plt.text(datetime.datetime(year, 4, 4, 0, 0), np.max(Spearman_cor)+10-length*0.28, text4)  
        plt.text(datetime.datetime(year, 4, 4, 0, 0), np.max(Spearman_cor)+10-length*0.34, text5)     
        
        #plt.text(3, 8,'a', bbox={'avg':avg_IC,'std':std_IC, 'np.min':np.min_IC,'avg/std':res})    
        plt.ylabel('Spearman Corr(%)')
        plt.title('Spearman IC(%)')
        plt.savefig(os.path.join(self.save_path, 'Spearman IC'))
        
        plt.show()
        return Spearman_cor


    def drawICDecay(self):
        all_spearman =  []
        date1 = pd.to_datetime(self.date,format = '%Y%m%d')
        for j in range(0,12):
            Spearman_cor=[]
            for i in range(0,len(date1)-12):
                thisweek = self.date[i]
                thisWeekData =self.factor_data.ix[thisweek]
                fwddate = self.date[i+j]            
                rtndata = self.fwrtn.ix[fwddate]
                stocks = list(set(rtndata.dropna().index) & set(thisWeekData.dropna().index))
                facs = thisWeekData.ix[stocks]
                fmrts = rtndata.ix[stocks]
                cor = stats.spearmanr(facs, fmrts).correlation
                Spearman_cor.append(cor)
            all_spearman.append(np.nanmean(Spearman_cor))
            
        plt.bar(range(1,13),np.array(all_spearman)*100)
        plt.ylabel('Spearman Corr %')
        plt.title('IC Decay')
        plt.savefig(os.path.join(self.save_path, 'IC Decay'))
        plt.show()
    
    
    def drawICDist(self,Spearman_cor):
        mean = np.nanmean(Spearman_cor/100.0)
        std = np.nanstd( Spearman_cor/100.0)
        a=list(pd.Series(Spearman_cor).dropna()/100.0)
        Y = norm(loc=mean,scale=std)
        t = np.arange(np.min(a)-0.2,np.max(a)+0.2,0.01)
        
        plt.subplots()    
        plt.hist(a,bins=40,normed=1, facecolor='blue', alpha=0.5)  
        plt.plot(t,Y.pdf(t),'r')
    
        plt.text(np.min(a)-0.15, 1.0/(math.sqrt(math.pi*2)*std) +0.3 , '$\mu$ = '+str(round(mean,2)))  
        plt.text(np.min(a)-0.15, 1.0/(math.sqrt(math.pi*2)*std) , '$\sigma$ = '+str(round(std,2))) 
        plt.text(np.min(a)-0.15, 1.0/(math.sqrt(math.pi*2)*std) -0.3, '$pval$ = '+str(round(stats.normaltest(a)[1],3)))  
        plt.xlabel('Spearman Corr')
        plt.title('Spearman IC distribution ')
        plt.savefig(os.path.join(self.save_path, 'Spearman IC distribution'))
        plt.show()
    def main(self,win,year,labels):
        self.drawDistrubution()
        self.calSerial(win,year)

        self.drawCoverage(win)
        self.drawTurnover(win)
        self.drawQuantileRtn()
        IR_all = self.calIR(self.mkt_index,win)
        SR_all = self.calSortinoRatio(self.mkt_index, win)
        self.drawWealthCurve(win, year)
        self.drawLogWealthCurve()
        self.drawSampleRtn(win, year)

        rtn_all = self.calAnualReturn(labels, win)
        std_all = self.calAnualVol(labels,win)
        
        result = pd.DataFrame([IR_all,SR_all,rtn_all,std_all],index = ["IR","SR","RTN","STD"],columns = labels)
        result.to_csv(os.path.join(self.save_path, "result.csv"))
        
        Spearman_cor = self.calSpearman(win,year)
        self.drawICDecay()
        self.drawICDist(Spearman_cor)