from typing import NamedTuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from typing import Callable
from scipy.stats import linregress 
raw_data = NamedTuple('raw_data', [('T', float), ('M', np.ndarray),('E', np.ndarray)])
result = NamedTuple('result', [("T",float),('avg', float), ('err', float)])

class fss:
    def __init__(self) -> None:
        self.raw :dict = {}
        self.processed = {}
        self.T_c = 0
        self.a_best = 0
        self.b_best = 0
        self.nu = 1

        self.logd = []
        self.binderd = []
        # The range 1 to 9 is chosen to compute moments up to the 9th order for logd and binderd calculations.
        for k_val in range(10):
            logd_k = lambda m, e, L, T,k=k_val: np.mean(m**k*e)/ np.mean(m**k) - np.mean(e)
            # <m^k *e >/ <m^k> - <e>
            self.logd.append(logd_k)

            binderd_k = lambda m, e, L, T,k = k_val: -(1/3)* (
                (np.mean(m**(2*k) * e) - np.mean(e) * np.mean(m**(2*k))) /
                    (np.mean(m**k)**2) 
                -2 * np.mean(m**(2*k)) 
                    * (np.mean(e * m**k) - np.mean(e) * np.mean(m**k)) /
                    np.mean(m**k)**3
            )
            # ((<e* m^2k> - <e>*<m^2k>) / <m^k>^2 - 2 <e* m^2k>(<e m^k>-<e>*<m^k>) / <m^k>^3)/T**2
            self.binderd.append(binderd_k)



    def add_raw_data(self,ext_e_data:np.ndarray,ext_m_data:np.ndarray,T:float,L:int,block_size:int = 20):
        """
        Add raw data to this object. 
        Args:
            data (np.ndarray): The raw data to be added.
            T (float): The temperature at which the data was collected.
            L (int): The system size.
            type (str): The type of data, either 'e' for energy or 'm' for magnetization.
        """
        if len(ext_e_data) != len(ext_m_data):
            raise ValueError("The length of energy and magnetization data must be the same.")
        if L not in self.raw.keys():
            self.raw[L] = []
        m_rtime = self._rtime(ext_m_data)
        e_rtime = self._rtime(ext_e_data)
        rtime = max(m_rtime, e_rtime)

        num = len(ext_e_data)

        ext_e_data = ext_e_data[:rtime * (num // rtime)]
        ext_m_data = ext_m_data[:rtime * (num // rtime)]

        ext_e_data = ext_e_data.reshape((num//rtime,rtime)) # reshape to (rtime, num/rtime). 
        ext_m_data = ext_m_data.reshape((num//rtime,rtime))

        ext_e_data = np.ascontiguousarray(ext_e_data, dtype=np.float64)
        ext_m_data = np.ascontiguousarray(ext_m_data, dtype=np.float64)

        self.raw[L].append(raw_data(T=T, M=ext_m_data, E=ext_e_data))

    def is_in(self,L:int,T:float)-> bool:
        """
        Check if the data for a specific system size and temperature is already stored.
        Args:
            L (int): The system size.
            T (float): The temperature.
        Returns:
            bool: True if the data is already stored, False otherwise.
        """
        if L in self.raw.keys():
            for entry in self.raw[L]:
                if abs(entry.T - T)<10**-5 :
                    return True
        return False
    
    def _sort_data(self):
        """
        Sort the data by temperature.
        """
        for L in self.raw.keys():
            self.raw[L].sort(key=lambda x: x.T)

    def see_all_data(self):
        """
        print all the data stored in this object.
        """
        for L, data in self.raw.items():
            print(f"Data for L={L}:")
            for i,entry in enumerate(data):
                print(f"T={round(entry.T,3)} with {np.size(entry.M,axis=1)} measurements at index {i}")

    def delete(self,L:int,ind:int):
        """
        Delete a specific entry from the stored data.
        """
        if L in self.raw:
            if ind < len(self.raw[L]):
                del self.raw[L][ind]
            else:
                raise IndexError("Index out of range for energy data.")
        if L in self.raw:
            if ind < len(self.raw[L]):
                del self.raw[L][ind]
            else:
                raise IndexError("Index out of range for magnetization data.")
        raise ValueError("No data found for the specified system size.")


    def _rtime(self, data:np.ndarray):
        N = len(data)
        data = data - np.mean(data)  # calculate variance around mean
        var = np.var(data, ddof=1)

        for search in range(100, N, N//100):
            acf = np.array([np.mean((data[:N-k]) * (data[k:])) for k in range(search)])
            if np.sum(acf < 0) > 0:
                break
        else:
            raise ValueError("No negative autocorrelation found in the data.")
        acf /= var
        k_max = np.argmax(acf < 0)
        if k_max == 0:
            k_max = len(acf)
        rtime = int(0.5 + np.sum(acf[1:k_max+1]))+1
        return 6*rtime 

    def ensemble_avg(self,f:Callable[[np.ndarray, np.ndarray, int, float], np.ndarray]):
        """Update one dictionary, containing the average and error of the data."""
        self.processed = {}

        for L ,data in self.raw.items():
            if L not in self.processed:
                self.processed[L] = []
            for (t,m,e) in data:
                # Jackknife resampling
                Q_jack = []
                for i in range(np.size(m,axis=1)):
                    m_jack = np.delete(m, i, axis=1)
                    e_jack = np.delete(e, i, axis=1)
                    Q_jack.append(f(m=m_jack, e=e_jack, L=L, T=t))
                Q_jack = np.array(Q_jack)
                Q_avg = np.mean(Q_jack)
                Q_err = np.sqrt((len(Q_jack) - 1) / len(Q_jack)) * np.std(Q_jack, ddof=1)
                self.processed[L].append(result(T=t, avg=Q_avg, err=Q_err))



    def _interpolate(self):
        """for each system size L, create an interpolation function for the processed data."""
        output = {}
        for L , data in self.processed.items():
            output[L] = interp1d(
                [x.T for x in data],
                [x.avg for x in data],
                kind='linear',
                bounds_error=False,
                fill_value=np.nan
            )
        return output

    def _str2lambda(self, expr:str)-> Callable:
        """
        convert a string to a lambda function
        example: <e*m> = np.mean(e*m,axis=0)
        """
        expr = expr.replace("<", "np.mean(")
        expr = expr.replace(">", ")")

        in_abs = False
        new_str = ""
        for letter in expr:
            if letter == "|":
                if in_abs:
                    new_str += ")"
                else:
                    new_str += "np.abs("
                in_abs = not in_abs
            else:
                new_str += letter

        return eval("lambda m,e,L,T: " + new_str, {"np": np})


    def plot(self,f:str | Callable,scale = False):
        if isinstance(f, str):
            f = self._str2lambda(f)
        self.ensemble_avg(f)
        for L, data in self.processed.items():
            if scale:
                T =( np.array([x.T for x in data]) - self.T_c)/self.T_c * L**self.b_best
                avg = np.array([x.avg for x in data])*L**self.a_best
                err = np.array([x.err for x in data])*L**self.a_best
                plt.errorbar(T, avg, yerr=err, label=f"L={L}", fmt='o', capsize=3,lw =1)
            else:
                T = [x.T for x in data]
                avg = [x.avg for x in data]
                err = [x.err for x in data]
                plt.errorbar(T, avg, yerr=err, label=f"L={L}", fmt='o', capsize=3,lw =1)
        return

    def fit_a(self,f:str|Callable):
        """
    Fit the data based on scaling hypothesis Q ~ L^{-a} f(t L^{1/nu}). So Q_max ~ L^{-a}.
        Args:
            f (Callable): The function to fit the data.
        """
        if isinstance(f, str):
            f = self._str2lambda(f)
        self.ensemble_avg(f)
        l_list = []
        y = []
        for L, data in self.processed.items():
            l_list.append(L)
            y.append(np.max([x.avg for x in data]))
        result = linregress(np.log(np.array(l_list)), np.log(np.array(y)))
        print(f"Fitted a: {result.slope:.3f} ± {result.stderr:.3f}")

        

    def fit_Tc(self,f):
        """
        Fit the critical temperature T_c based on existing nu. Fit the data by Tc(L) = T_c + A* L^(-1/nu)
        Args:
            f (Callable): The function to fit the data.
        """
        if isinstance(f, str):
            f = self._str2lambda(f)
        self.ensemble_avg(f)
        l_list = []
        y = []
        for L, data in self.processed.items():
            T = np.array([x.T for x in data])
            avg = np.array([x.avg for x in data])
            max_index = np.argmax(avg)
            l_list.append(L)
            y.append(T[max_index])
        result = linregress(np.array(l_list)**(-1/self.nu), np.array(y))
        print(f"Fitted T_c: {result.intercept:.3f} ± {result.intercept_stderr:.3f}")
    

