import numpy as np
import pylab as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
import time

class AutonomousSystem2D():
    """
    solver for 2d autonomous systems 
    with x,y dependent and t independent variables
    
    Linear system
    dxdt = ax-y
    dydt = e(bx-y)
    parameters: 'a', 'b', 'eps'
    """
    xvar_name = 'x' # first dependent variable
    yvar_name = 'y' # second dependent variable
    tvar_name = 't' # independent variable
    
    # parameters for plotting results
    y_min, y_max = -2,2
    x_min, x_max = -2,2
    
    fig = None
    ax1 = None
    ax2 = None
    canvas = None
    
    def __init__(self, x0, y0, parameters, run_time=10, time_step=0.01, t0=0., front_end=True):
        
        # dictionary with model parameters
        self.parameters = parameters
        
        self.x0 = x0
        self.y0 = y0
        
        self.x = None 
        self.y = None 
        
        # The time to integrate over
        self.run_time = run_time
        self.t0 = t0
        self.time_step = time_step
        self.t = np.arange(self.t0, self.run_time, self.time_step)
        self.n_steps = len(self.t)
        self.front_end = front_end # False for webapps and true if figure gets displayed
        
    @staticmethod
    def dxdt(x, y, self):
        return  self.parameters['a'] * x - y
    
    @staticmethod
    def dydt(x, y, self): 
        return self.parameters['eps'] * (self.parameters['b'] * x - y) 
        
    @staticmethod
    def dXdt(X, t, self):
        """
        Integrate

        |  :param X: [x, y] dependent variables
        |  :param t: time
        |  :return: calculate membrane potential & activation variables
        """
        x, y = X

        
        dxdt = self.dxdt(x, y, self)
        dydt = self.dydt(x, y, self)
        
        return dxdt, dydt
    
    def fixpunt_function(self,xy):
        x,y = xy
        dxdt = self.dxdt(x, y, self)
        dydt = self.dydt(x, y, self)
        return dxdt, dydt
    
    def solve(self):
        X = odeint(self.dXdt, [self.x0, self.y0], self.t, args=(self,))
        self.x = X[:,0]
        self.y = X[:,1]
             
    def run(self):
        self.solve()
        self.plot_solution()
        
    def nullcline_dx(self, x):
        return self.parameters['a']* x 
    
    def nullcline_dy(self, x):
        return x * self.parameters['b'] 
    
    
    def plot_solution(self, ):

        x_min, x_max, y_min, y_max = self.get_plot_lims()
        
        if self.front_end:
            self.fig = plt.figure(figsize=(7,3.5), dpi=150)
        else:
            self.fig = Figure(figsize=(7,3.5), dpi=150)
        
        ax1 = self.fig.add_subplot(121)
        ax1.plot(self.t,self.x, c='navy', label=self.xvar_name)
        ax1.plot(self.t,self.y, c='firebrick', label=self.yvar_name)
        ax1.legend(fontsize=10)
        ax1.set_xlabel(self.tvar_name)
        ax1.set_ylabel('{x}({t}), {y}({t})'.format(x=self.xvar_name, y=self.yvar_name, t=self.tvar_name))
        
        ax2 = self.fig.add_subplot(122)
        ax2.set_xlabel(self.xvar_name)
        ax2.set_ylabel(self.yvar_name)
        ax2.plot(self.x, self.y, 'k-')
        ax2.plot(self.x[0], self.y[0], 'ro')
        
        # add vector field
        xx,yy = np.meshgrid(np.linspace(x_min,x_max,15),\
                  np.linspace(y_min,y_max,15)
                  )
        dx, dy = self.dxdt(xx,yy,self), self.dydt(xx,yy,self)
        ax2.quiver(xx,yy,dx,dy)
        
        x_grid = np.linspace(x_min, x_max, 1000)
        nullcline_dx_vals = self.nullcline_dx(x_grid)
        nullcline_dy_vals = self.nullcline_dy(x_grid)
        ax2.plot(x_grid, nullcline_dx_vals, c='c')
        ax2.plot(x_grid, nullcline_dy_vals, c='y')
        ax2.set_ylim(y_min, y_max)
        ax2.set_xlim(x_min, x_max)
        
        
        if self.front_end:
            plt.tight_layout()
            plt.show()   
        else:
            self.canvas = FigureCanvasAgg(self.fig)
    
    
    def get_plot_lims(self, ):
        xmin, xmax = np.min(self.x), np.max(self.x)
        ymin, ymax = np.min(self.y), np.max(self.y)
        xrange = xmax - xmin
        yrange = ymax - ymin
        m = 0.05
        xlims = (xmin - m * xrange, xmax + m * xrange)
        ylims = (ymin - m * yrange, ymax + m * yrange)
        return xlims[0], xlims[1], ylims[0], ylims[1]
    
    def update(self, n):
        """
        update plot in animations
        """
        ax1 = self.ax1
        ax2 = self.ax2
        
        if ax1.lines:     
            x_line = ax1.lines[0]
            x_line.set_xdata(self.t[:n])
            x_line.set_ydata(self.x[:n])
            y_line = ax1.lines[1]
            y_line.set_xdata(self.t[:n])
            y_line.set_ydata(self.y[:n])    
            l3 =ax2.lines[2]
            l3.set_xdata(self.x[:n])
            l3.set_ydata(self.y[:n])
            dot = ax2.lines[3]
            dot.set_xdata(self.x[n-2:n])
            dot.set_ydata(self.y[n-2:n])
        self.fig.canvas.draw()   
        #self.canvas = FigureCanvasAgg(self.fig)
        time.sleep(0.03)
        
    def webapp_animation(self,equal_range=False, n_frames=100 ):
        x_min, x_max, y_min, y_max = self.get_plot_lims()
        if equal_range:
            x_min = min(x_min, y_min)
            y_min = x_min
            x_max = max(x_max, y_max)
            y_max = x_max
        
        xx,yy = np.meshgrid(np.linspace(x_min,x_max,15),\
                  np.linspace(y_min,y_max,15)
                  )
        dx, dy = self.dxdt(xx,yy,self), self.dydt(xx,yy,self)
        x_grid = np.linspace(x_min, x_max, 1000)
        
        nullcline_dx_vals = self.nullcline_dx(x_grid)
        nullcline_dy_vals = self.nullcline_dy(x_grid)
        
        # Your animation generation code here
        self.fig = plt.figure(figsize=(5,2.5), dpi=150)
        self.ax1 = self.fig.add_subplot(121)    
        self.ax2 = self.fig.add_subplot(122)
        # Create your animation using FuncAnimation
        # Example:
        def animate(frame, step=50):
            print("frame", frame, " of ",n_frames)
            # Update the plot for each frame
            #fig.clear()  # Clear the current figure
            self.ax1.clear()
            self.ax2.clear()
       
            # Plot the updated data
            self.ax1.set_ylim(y_min, y_max)
            self.ax1.set_xlim(0,self.run_time)
            self.ax1.set_xlabel(self.tvar_name)
            self.ax1.set_ylabel('{x}({t}), {y}({t})'.format(x=self.xvar_name, y=self.yvar_name, t=self.tvar_name))    
            
            self.ax2.quiver(xx,yy,dx,dy)
            self.ax2.plot(x_grid, nullcline_dx_vals, c='c')
            self.ax2.plot(x_grid, nullcline_dy_vals, c='y')
            self.ax2.set_ylim(y_min, y_max)
            self.ax2.set_xlim(x_min, x_max)
            self.ax2.set_xlabel(self.xvar_name)
            self.ax2.set_ylabel(self.yvar_name)

            self.ax1.plot(self.t[:frame*step],self.x[:frame*step], c='navy', label='v')
            self.ax1.plot(self.t[:frame*step],self.y[:frame*step], c='firebrick', label='w')
            self.ax2.plot(self.x[:frame*step], self.y[:frame*step], 'k-')
            n = max(frame*step,2)
            self.ax2.plot(self.x[n-2:n], self.y[n-2:n], 'r-', lw=4)
            
            self.ax1.legend(fontsize=6, loc='upper right')
            plt.tight_layout()

        ani = FuncAnimation(self.fig, animate, frames=n_frames, interval=250)
        # interval is delay between frames in ms
        return ani
    
    def notebook_animation(self,equal_range=False):
        #%matplotlib notebook

        x_min, x_max, y_min, y_max = self.get_plot_lims()
        if equal_range:
            x_min = min(x_min, y_min)
            y_min = x_min
            x_max = max(x_max, y_max)
            y_max = x_max

        self.fig = plt.figure(figsize=(5,2.5), dpi=150)

        
        self.ax1 = self.fig.add_subplot(121)
        self.ax1.set_ylim(min(y_min,x_min), max(y_max,x_max))
        self.ax1.set_xlim(0,self.run_time)
        self.ax1.set_xlabel(self.tvar_name)
        
        self.ax2 = self.fig.add_subplot(122)
        self.ax2.set_xlabel(self.xvar_name)
        self.ax2.set_ylabel(self.yvar_name)
        # add vector field
        xx,yy = np.meshgrid(np.linspace(x_min,x_max,15),\
                  np.linspace(y_min,y_max,15)
                  )
        dx, dy = self.dxdt(xx,yy,self), self.dydt(xx,yy,self)
        self.ax2.quiver(xx,yy,dx,dy)
        
        x_grid = np.linspace(x_min, x_max, 1000)
        nullcline_dx_vals = self.nullcline_dx(x_grid)
        nullcline_dy_vals = self.nullcline_dy(x_grid)
        self.ax2.plot(x_grid, nullcline_dx_vals, c='c')
        self.ax2.plot(x_grid, nullcline_dy_vals, c='y')
        self.ax2.set_ylim(y_min, y_max)
        self.ax2.set_xlim(x_min, x_max)
        
        plt.tight_layout()
        
        self.ax1.plot(self.t[:10],self.x[:10], c='navy', label='v')
        self.ax1.plot(self.t[:10],self.y[:10], c='firebrick', label='w')
        self.ax2.plot(self.x[:10], self.y[:10], 'k-')
        self.ax2.plot(self.x[:2], self.y[:2], 'r-', lw=4)
        self.ax1.legend(fontsize=6)
        for i in range(20,len(self.t),20):
            self.update(i)

LinearSystem = AutonomousSystem2D         
   
class StationaryDiffusion1D(AutonomousSystem2D):
    
    xvar_name = 'x'#'c' # concentration
    yvar_name = 'y'#'j' # current
    tvar_name = 't'#'x' # space
    
    @staticmethod
    def dxdt(x, y, self):
        return  -y / self.parameters['D']
    
    @staticmethod
    def dydt(x, y, self):
        return 0.
    
    def nullcline_dx(self, x):
        return 0. * x
    
    def nullcline_dy(self, x):
        return np.nan * x
    
class FitzHugh_Nagumo(AutonomousSystem2D):
    
    xvar_name = 'v'
    yvar_name = 'w'
    tvar_name = 't'
    
    @staticmethod
    def dxdt(x, y, self):
        return  x - np.power(x,3)/3. - y + self.parameters['i']
    
    @staticmethod
    def dydt(x, y, self):
        return x/self.parameters['tau'] + self.parameters['a']/self.parameters['tau'] - self.parameters['b']/self.parameters['tau'] * y 

    def nullcline_dx(self, x):
        return x - np.power(x,3)/3. + self.parameters['i']
    
    def nullcline_dy(self, x):
        return (x + self.parameters['a']) / self.parameters['b'] 
    
    def jacobi(self,x,y):
        return np.array([[1.-x**2, -1.],
                         [1./self.parameters['tau'], -self.parameters['b']/self.parameters['tau']]])
    
    def trace(self, x,y):
        return 1. - x**2 -self.parameters['b']/self.parameters['tau']

    def det(self, x,y):
        return (1.-self.parameters['b'])/self.parameters['tau'] + x**2*self.parameters['b']/self.parameters['tau']
    
class Genetic_control(AutonomousSystem2D):
    
    xvar_name = 'x'
    yvar_name = 'y'
    tvar_name = 't'
    
    y_min, y_max = -0.5,1.5
    x_min, x_max = -0.2,1.8
    
    @staticmethod
    def dxdt(x, y, self):
        return  - self.parameters['a'] * x + y
    
    @staticmethod
    def dydt(x, y, self):
        return x**2 / (1+x**2) - self.parameters['b'] * y
    
    def nullcline_dx(self, x):
        return self.parameters['a'] * x
    
    def nullcline_dy(self, x):
        return x**2 /self.parameters['b'] /(1+x**2)
    
    
class Lotka_volterra(AutonomousSystem2D):
    
    xvar_name = 'predator population'
    yvar_name = 'prey population'
    tvar_name = 't' # time
    
    y_min, y_max = -1,20
    x_min, x_max = -1,70
    
    @staticmethod
    def dxdt(x, y, self):
        return  self.parameters['a']*x-self.parameters['b']*x*y
    
    @staticmethod
    def dydt(x, y, self):
        return self.parameters['c']*x*y-self.parameters['d']*y
    
    def nullcline_dx(self, x):
        return np.nan * x
    
    def nullcline_dy(self, x):
        return np.nan * x
    
class DampedHarmonicOscillator(AutonomousSystem2D):
    
    y_min, y_max = -2,2
    x_min, x_max = -2,2
    
    @staticmethod
    def dxdt(x, y, self):
        return y
    
    @staticmethod
    def dydt(x, y, self):
        return -2 *  self.parameters['gamma'] * y - self.parameters['omega']**2* x
    
    def nullcline_dx(self, x):
        return x
    
    def nullcline_dy(self, x):
        return - self.parameters['omega']**2 * x / (2 *  self.parameters['gamma'])