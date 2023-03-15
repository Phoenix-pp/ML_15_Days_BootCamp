#!/usr/bin/env python
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Introduction-to-visualization-in-Python-using" data-toc-modified-id="Introduction-to-visualization-in-Python-using-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction to visualization in Python using</a></div><div class="lev1 toc-item"><a href="#Why-matplotlib?" data-toc-modified-id="Why-matplotlib?-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Why matplotlib?</a></div><div class="lev2 toc-item"><a href="#Introduction" data-toc-modified-id="Introduction-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Introduction</a></div><div class="lev2 toc-item"><a href="#MATLAB-like-API" data-toc-modified-id="MATLAB-like-API-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>MATLAB-like API</a></div><div class="lev3 toc-item"><a href="#Example" data-toc-modified-id="Example-2.2.1"><span class="toc-item-num">2.2.1&nbsp;&nbsp;</span>Example</a></div><div class="lev2 toc-item"><a href="#The-matplotlib-object-oriented-API" data-toc-modified-id="The-matplotlib-object-oriented-API-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>The matplotlib object-oriented API</a></div><div class="lev3 toc-item"><a href="#Figure-size,-aspect-ratio-and-DPI" data-toc-modified-id="Figure-size,-aspect-ratio-and-DPI-2.3.1"><span class="toc-item-num">2.3.1&nbsp;&nbsp;</span>Figure size, aspect ratio and DPI</a></div><div class="lev3 toc-item"><a href="#Saving-figures" data-toc-modified-id="Saving-figures-2.3.2"><span class="toc-item-num">2.3.2&nbsp;&nbsp;</span>Saving figures</a></div><div class="lev4 toc-item"><a href="#What-formats-are-available-and-which-ones-should-be-used-for-best-quality?" data-toc-modified-id="What-formats-are-available-and-which-ones-should-be-used-for-best-quality?-2.3.2.1"><span class="toc-item-num">2.3.2.1&nbsp;&nbsp;</span>What formats are available and which ones should be used for best quality?</a></div><div class="lev3 toc-item"><a href="#Legends,-labels-and-titles" data-toc-modified-id="Legends,-labels-and-titles-2.3.3"><span class="toc-item-num">2.3.3&nbsp;&nbsp;</span>Legends, labels and titles</a></div><div class="lev3 toc-item"><a href="#Formatting-text:-LaTeX,-fontsize,-font-family" data-toc-modified-id="Formatting-text:-LaTeX,-fontsize,-font-family-2.3.4"><span class="toc-item-num">2.3.4&nbsp;&nbsp;</span>Formatting text: LaTeX, fontsize, font family</a></div><div class="lev3 toc-item"><a href="#Setting-colors,-linewidths,-linetypes" data-toc-modified-id="Setting-colors,-linewidths,-linetypes-2.3.5"><span class="toc-item-num">2.3.5&nbsp;&nbsp;</span>Setting colors, linewidths, linetypes</a></div><div class="lev4 toc-item"><a href="#Colors" data-toc-modified-id="Colors-2.3.5.1"><span class="toc-item-num">2.3.5.1&nbsp;&nbsp;</span>Colors</a></div><div class="lev4 toc-item"><a href="#Line-and-marker-styles" data-toc-modified-id="Line-and-marker-styles-2.3.5.2"><span class="toc-item-num">2.3.5.2&nbsp;&nbsp;</span>Line and marker styles</a></div><div class="lev3 toc-item"><a href="#Control-over-axis-appearance" data-toc-modified-id="Control-over-axis-appearance-2.3.6"><span class="toc-item-num">2.3.6&nbsp;&nbsp;</span>Control over axis appearance</a></div><div class="lev4 toc-item"><a href="#Plot-range" data-toc-modified-id="Plot-range-2.3.6.1"><span class="toc-item-num">2.3.6.1&nbsp;&nbsp;</span>Plot range</a></div><div class="lev4 toc-item"><a href="#Logarithmic-scale" data-toc-modified-id="Logarithmic-scale-2.3.6.2"><span class="toc-item-num">2.3.6.2&nbsp;&nbsp;</span>Logarithmic scale</a></div><div class="lev3 toc-item"><a href="#Placement-of-ticks-and-custom-tick-labels" data-toc-modified-id="Placement-of-ticks-and-custom-tick-labels-2.3.7"><span class="toc-item-num">2.3.7&nbsp;&nbsp;</span>Placement of ticks and custom tick labels</a></div><div class="lev4 toc-item"><a href="#Scientific-notation" data-toc-modified-id="Scientific-notation-2.3.7.1"><span class="toc-item-num">2.3.7.1&nbsp;&nbsp;</span>Scientific notation</a></div><div class="lev3 toc-item"><a href="#Axis-number-and-axis-label-spacing" data-toc-modified-id="Axis-number-and-axis-label-spacing-2.3.8"><span class="toc-item-num">2.3.8&nbsp;&nbsp;</span>Axis number and axis label spacing</a></div><div class="lev4 toc-item"><a href="#Axis-position-adjustments" data-toc-modified-id="Axis-position-adjustments-2.3.8.1"><span class="toc-item-num">2.3.8.1&nbsp;&nbsp;</span>Axis position adjustments</a></div><div class="lev3 toc-item"><a href="#Axis-grid" data-toc-modified-id="Axis-grid-2.3.9"><span class="toc-item-num">2.3.9&nbsp;&nbsp;</span>Axis grid</a></div><div class="lev3 toc-item"><a href="#Axis-spines" data-toc-modified-id="Axis-spines-2.3.10"><span class="toc-item-num">2.3.10&nbsp;&nbsp;</span>Axis spines</a></div><div class="lev3 toc-item"><a href="#Twin-axes" data-toc-modified-id="Twin-axes-2.3.11"><span class="toc-item-num">2.3.11&nbsp;&nbsp;</span>Twin axes</a></div><div class="lev3 toc-item"><a href="#Axes-where-x-and-y-is-zero" data-toc-modified-id="Axes-where-x-and-y-is-zero-2.3.12"><span class="toc-item-num">2.3.12&nbsp;&nbsp;</span>Axes where x and y is zero</a></div><div class="lev2 toc-item"><a href="#Other-2D-plot-styles" data-toc-modified-id="Other-2D-plot-styles-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Other 2D plot styles</a></div><div class="lev3 toc-item"><a href="#Text-annotation" data-toc-modified-id="Text-annotation-2.4.1"><span class="toc-item-num">2.4.1&nbsp;&nbsp;</span>Text annotation</a></div><div class="lev3 toc-item"><a href="#Figures-with-multiple-subplots-and-insets" data-toc-modified-id="Figures-with-multiple-subplots-and-insets-2.4.2"><span class="toc-item-num">2.4.2&nbsp;&nbsp;</span>Figures with multiple subplots and insets</a></div><div class="lev4 toc-item"><a href="#subplots" data-toc-modified-id="subplots-2.4.2.1"><span class="toc-item-num">2.4.2.1&nbsp;&nbsp;</span>subplots</a></div><div class="lev4 toc-item"><a href="#subplot2grid" data-toc-modified-id="subplot2grid-2.4.2.2"><span class="toc-item-num">2.4.2.2&nbsp;&nbsp;</span>subplot2grid</a></div><div class="lev4 toc-item"><a href="#gridspec" data-toc-modified-id="gridspec-2.4.2.3"><span class="toc-item-num">2.4.2.3&nbsp;&nbsp;</span>gridspec</a></div><div class="lev4 toc-item"><a href="#add_axes" data-toc-modified-id="add_axes-2.4.2.4"><span class="toc-item-num">2.4.2.4&nbsp;&nbsp;</span>add_axes</a></div><div class="lev3 toc-item"><a href="#Colormap-and-contour-figures" data-toc-modified-id="Colormap-and-contour-figures-2.4.3"><span class="toc-item-num">2.4.3&nbsp;&nbsp;</span>Colormap and contour figures</a></div><div class="lev4 toc-item"><a href="#pcolor" data-toc-modified-id="pcolor-2.4.3.1"><span class="toc-item-num">2.4.3.1&nbsp;&nbsp;</span>pcolor</a></div><div class="lev4 toc-item"><a href="#imshow" data-toc-modified-id="imshow-2.4.3.2"><span class="toc-item-num">2.4.3.2&nbsp;&nbsp;</span>imshow</a></div><div class="lev4 toc-item"><a href="#contour" data-toc-modified-id="contour-2.4.3.3"><span class="toc-item-num">2.4.3.3&nbsp;&nbsp;</span>contour</a></div><div class="lev2 toc-item"><a href="#3D-figures" data-toc-modified-id="3D-figures-2.5"><span class="toc-item-num">2.5&nbsp;&nbsp;</span>3D figures</a></div><div class="lev4 toc-item"><a href="#Surface-plots" data-toc-modified-id="Surface-plots-2.5.0.1"><span class="toc-item-num">2.5.0.1&nbsp;&nbsp;</span>Surface plots</a></div><div class="lev4 toc-item"><a href="#Wire-frame-plot" data-toc-modified-id="Wire-frame-plot-2.5.0.2"><span class="toc-item-num">2.5.0.2&nbsp;&nbsp;</span>Wire-frame plot</a></div><div class="lev4 toc-item"><a href="#Coutour-plots-with-projections" data-toc-modified-id="Coutour-plots-with-projections-2.5.0.3"><span class="toc-item-num">2.5.0.3&nbsp;&nbsp;</span>Coutour plots with projections</a></div><div class="lev4 toc-item"><a href="#Change-the-view-angle" data-toc-modified-id="Change-the-view-angle-2.5.0.4"><span class="toc-item-num">2.5.0.4&nbsp;&nbsp;</span>Change the view angle</a></div><div class="lev3 toc-item"><a href="#Animations" data-toc-modified-id="Animations-2.5.1"><span class="toc-item-num">2.5.1&nbsp;&nbsp;</span>Animations</a></div><div class="lev3 toc-item"><a href="#Backends" data-toc-modified-id="Backends-2.5.2"><span class="toc-item-num">2.5.2&nbsp;&nbsp;</span>Backends</a></div><div class="lev4 toc-item"><a href="#Generating-SVG-with-the-svg-backend" data-toc-modified-id="Generating-SVG-with-the-svg-backend-2.5.2.1"><span class="toc-item-num">2.5.2.1&nbsp;&nbsp;</span>Generating SVG with the svg backend</a></div><div class="lev4 toc-item"><a href="#The-IPython-notebook-inline-backend" data-toc-modified-id="The-IPython-notebook-inline-backend-2.5.2.2"><span class="toc-item-num">2.5.2.2&nbsp;&nbsp;</span>The IPython notebook inline backend</a></div><div class="lev4 toc-item"><a href="#Interactive-backend-(this-makes-more-sense-in-a-python-script-file)" data-toc-modified-id="Interactive-backend-(this-makes-more-sense-in-a-python-script-file)-2.5.2.3"><span class="toc-item-num">2.5.2.3&nbsp;&nbsp;</span>Interactive backend (this makes more sense in a python script file)</a></div><div class="lev1 toc-item"><a href="#Other-visualization-packages" data-toc-modified-id="Other-visualization-packages-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Other visualization packages</a></div><div class="lev2 toc-item"><a href="#Further-reading" data-toc-modified-id="Further-reading-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Further reading</a></div>

# # Introduction to visualization in Python using
# 
# <img src="http://matplotlib.org/_static/logo2.svg">

# In[1]:


from IPython.display import Image, HTML, display
from glob import glob

Links=[]
Links.append("http://matplotlib.org/_images/ellipse_demo1.png") 
Links.append("http://matplotlib.org/_images/finance_work21.png")
Links.append("http://matplotlib.org/_images/eeg_small.png")
  
imagesList=''.join( ["<img style='width: 300px; margin: 0px; float: left; border: 1px solid black;' src='%s' />" % str(s) 
                     for s in Links ])
display(HTML(imagesList))


# # Why matplotlib?
# 
# * Comes intalled with anaconda
#    * Settings can be customised by editing ~/.matplotlib/matplotlibrc
#    * User friendly, but powerful, plotting capabilites for python http://matplotlib.sourceforge.net/
#        * Helpful website: many examples
# 
# 
# 

# In this notebook, we will explore 2D and 3D plotting using matplotlib package of Python.
# 
# **We will also demonstrate how to create multiple document formats using this notebook content. Typically we will render the notebook in HTML, LaTex, rst and markdown format** 

# * You can use matplotlib inside your jupyter notebook by calling *%matplotlib inline* , which has the advantage of keeping your plots in one place. 
# 
#     If you're having trouble running matplotlib, [here](http://stackoverflow.com/questions/19410042/how-to-make-ipython-
#     notebook-matplotlib-plot-inline) are a few common solutions. 
#     
# **What %matplotlib inline does?**
# 
# * %matplotlib inline activates the inline backend and calls images as static pngs. 
# * A new option--%matplotlib notebook--lets you interact with the plot in a Notebook. This works in IPython/jupyter 3.x or newer notebooks; for older IPython or jupyter versions, use %matplotlib nbagg.
# * In our case **we will use %matplotlib notebook**

# In[2]:


# This line configures matplotlib to show figures embedded in the notebook, 
# instead of opening a new window for each figure. More about that later. 
# If you are using an old version of IPython, try using '%pylab inline' instead.
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Introduction

# Matplotlib is an excellent 2D and 3D graphics library for generating scientific figures. Some of the many advantages of this library include:
# 
# * Easy to get started
# * Support for $\LaTeX$ formatted labels and texts
# * Great control of every element in a figure, including figure size and DPI. 
# * High-quality output in many formats, including PNG, PDF, SVG, EPS, and PGF.
# * GUI for interactively exploring figures *and* support for headless generation of figure files (useful for batch jobs).
# 
# One of the key features of matplotlib that I would like to emphasize, and that I think makes matplotlib highly suitable for generating figures for scientific publications is that all aspects of the figure can be controlled *programmatically*. This is important for reproducibility and convenient when one needs to regenerate the figure with updated data or change its appearance. 
# 
# More information at the Matplotlib web page: http://matplotlib.org/

# To get started using Matplotlib in a Python program, either include the symbols from the `pylab` module (the easy way):

# In[3]:


from pylab import *


# or import the `matplotlib.pyplot` module under the name `plt` (the tidy way):

# In[4]:


import matplotlib
import matplotlib.pyplot as plt


# In[5]:


import numpy as np


# ## MATLAB-like API

# The easiest way to get started with plotting using matplotlib is often to use the MATLAB-like API provided by matplotlib. 
# 
# It is designed to be compatible with MATLAB's plotting functions, so it is easy to get started with if you are familiar with MATLAB.
# 
# To use this API from matplotlib, we need to include the symbols in the `pylab` module: 

# In[6]:


from pylab import *


# ### Example

# A simple figure with MATLAB-like plotting API:

# In[7]:


x = np.linspace(0, 5, 10)
y = x ** 2


# In[8]:


figure()
plot(x, y, 'r')
xlabel('x')
ylabel('y')
title('title')
show()


# Most of the plotting related functions in MATLAB are covered by the `pylab` module. For example, subplot and color/symbol selection:

# In[9]:


subplot(1,2,1)
plot(x, y, 'r--')
subplot(1,2,2)
plot(y, x, 'g*-');


# The good thing about the pylab MATLAB-style API is that it is easy to get started with if you are familiar with MATLAB, and it has a minumum of coding overhead for simple plots. 
# 
# However, I'd encourrage not using the MATLAB compatible API for anything but the simplest figures.
# 
# Instead, I recommend learning and using matplotlib's object-oriented plotting API. It is remarkably powerful. For advanced figures with subplots, insets and other components it is very nice to work with. 

# ## The matplotlib object-oriented API

# The main idea with object-oriented programming is to have objects that one can apply functions and actions on, and no object or program states should be global (such as the MATLAB-like API). The real advantage of this approach becomes apparent when more than one figure is created, or when a figure contains more than one subplot. 
# 
# To use the object-oriented API we start out very much like in the previous example, but instead of creating a new global figure instance we store a reference to the newly created figure instance in the `fig` variable, and from it we create a new axis instance `axes` using the `add_axes` method in the `Figure` class instance `fig`:

# In[10]:


fig = plt.figure()

axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)

axes.plot(x, y, 'r')

axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('title');


# Although a little bit more code is involved, the advantage is that we now have full control of where the plot axes are placed, and we can easily add more than one axis to the figure:

# In[11]:


fig = plt.figure()

axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3]) # inset axes

# main figure
axes1.plot(x, y, 'r')
axes1.set_xlabel('x')
axes1.set_ylabel('y')
axes1.set_title('title')

# insert
axes2.plot(y, x, 'g')
axes2.set_xlabel('y')
axes2.set_ylabel('x')
axes2.set_title('insert title');


# If we don't care about being explicit about where our plot axes are placed in the figure canvas, then we can use one of the many axis layout managers in matplotlib. My favorite is `subplots`, which can be used like this:

# In[12]:


fig, axes = plt.subplots()

axes.plot(x, y, 'r')
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('title');


# In[13]:


fig, axes = plt.subplots(nrows=1, ncols=2)

for ax in axes:
    ax.plot(x, y, 'r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('title')


# That was easy, but it isn't so pretty with overlapping figure axes and labels, right?
# 
# We can deal with that by using the `fig.tight_layout` method, which automatically adjusts the positions of the axes on the figure canvas so that there is no overlapping content:

# In[14]:


fig, axes = plt.subplots(nrows=1, ncols=2)

for ax in axes:
    ax.plot(x, y, 'r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('title')
    
fig.tight_layout()


# ### Figure size, aspect ratio and DPI

# Matplotlib allows the aspect ratio, DPI and figure size to be specified when the `Figure` object is created, using the `figsize` and `dpi` keyword arguments. `figsize` is a tuple of the width and height of the figure in inches, and `dpi` is the dots-per-inch (pixel per inch). To create an 800x400 pixel, 100 dots-per-inch figure, we can do: 

# In[15]:


fig = plt.figure(figsize=(8,4), dpi=100)


# The same arguments can also be passed to layout managers, such as the `subplots` function:

# In[16]:


fig, axes = plt.subplots(figsize=(12,3))

axes.plot(x, y, 'r')
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('title');


# ### Saving figures

# To save a figure to a file we can use the `savefig` method in the `Figure` class:

# In[17]:


fig.savefig("filename.png")


# Here we can also optionally specify the DPI and choose between different output formats:

# In[18]:


fig.savefig("filename.png", dpi=200)


# #### What formats are available and which ones should be used for best quality?

# Matplotlib can generate high-quality output in a number formats, including PNG, JPG, EPS, SVG, PGF and PDF. For scientific papers, I recommend using PDF whenever possible. (LaTeX documents compiled with `pdflatex` can include PDFs using the `includegraphics` command). In some cases, PGF can also be good alternative.

# ### Legends, labels and titles

# Now that we have covered the basics of how to create a figure canvas and add axes instances to the canvas, let's look at how decorate a figure with titles, axis labels, and legends.

# **Figure titles**
# 
# A title can be added to each axis instance in a figure. To set the title, use the `set_title` method in the axes instance:

# In[19]:


ax.set_title("title");


# **Axis labels**
# 
# Similarly, with the methods `set_xlabel` and `set_ylabel`, we can set the labels of the X and Y axes:

# In[20]:


ax.set_xlabel("x")
ax.set_ylabel("y");


# **Legends**
# 
# Legends for curves in a figure can be added in two ways. One method is to use the `legend` method of the axis object and pass a list/tuple of legend texts for the previously defined curves:

# In[21]:


ax.legend(["curve1", "curve2", "curve3"]);


# The method described above follows the MATLAB API. It is somewhat prone to errors and unflexible if curves are added to or removed from the figure (resulting in a wrongly labelled curve).
# 
# A better method is to use the `label="label text"` keyword argument when plots or other objects are added to the figure, and then using the `legend` method without arguments to add the legend to the figure: 

# In[22]:


ax.plot(x, x**2, label="curve1")
ax.plot(x, x**3, label="curve2")
ax.legend();


# The advantage with this method is that if curves are added or removed from the figure, the legend is automatically updated accordingly.
# 
# The `legend` function takes an optional keyword argument `loc` that can be used to specify where in the figure the legend is to be drawn. The allowed values of `loc` are numerical codes for the various places the legend can be drawn. See http://matplotlib.org/users/legend_guide.html#legend-location for details. Some of the most common `loc` values are:

# In[23]:


ax.legend(loc=0) # let matplotlib decide the optimal location
ax.legend(loc=1) # upper right corner
ax.legend(loc=2) # upper left corner
ax.legend(loc=3) # lower left corner
ax.legend(loc=4) # lower right corner
# .. many more options are available


# The following figure shows how to use the figure title, axis labels and legends described above:

# In[24]:


fig, ax = plt.subplots()

ax.plot(x, x**2, label="y = x**2")
ax.plot(x, x**3, label="y = x**3")
ax.legend(loc=2); # upper left corner
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('title');


# ### Formatting text: LaTeX, fontsize, font family

# The figure above is functional, but it does not (yet) satisfy the criteria for a figure used in a publication. First and foremost, we need to have LaTeX formatted text, and second, we need to be able to adjust the font size to appear right in a publication.
# 
# Matplotlib has great support for LaTeX. All we need to do is to use dollar signs encapsulate LaTeX in any text (legend, title, label, etc.). For example, `"$y=x^3$"`.
# 
# But here we can run into a slightly subtle problem with LaTeX code and Python text strings. In LaTeX, we frequently use the backslash in commands, for example `\alpha` to produce the symbol $\alpha$. But the backslash already has a meaning in Python strings (the escape code character). To avoid Python messing up our latex code, we need to use "raw" text strings. Raw text strings are prepended with an '`r`', like `r"\alpha"` or `r'\alpha'` instead of `"\alpha"` or `'\alpha'`:

# In[25]:


fig, ax = plt.subplots()

ax.plot(x, x**2, label=r"$y = \alpha^2$")
ax.plot(x, x**3, label=r"$y = \alpha^3$")
ax.legend(loc=2) # upper left corner
ax.set_xlabel(r'$\alpha$', fontsize=18)
ax.set_ylabel(r'$y$', fontsize=18)
ax.set_title('title');


# We can also change the global font size and font family, which applies to all text elements in a figure (tick labels, axis labels and titles, legends, etc.):

# In[26]:


# Update the matplotlib configuration parameters:
matplotlib.rcParams.update({'font.size': 18, 'font.family': 'serif'})


# In[27]:


fig, ax = plt.subplots()

ax.plot(x, x**2, label=r"$y = \alpha^2$")
ax.plot(x, x**3, label=r"$y = \alpha^3$")
ax.legend(loc=2) # upper left corner
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$y$')
ax.set_title('title');


# A good choice of global fonts are the STIX fonts: 

# In[28]:


# Update the matplotlib configuration parameters:
matplotlib.rcParams.update({'font.size': 18, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})


# In[29]:


fig, ax = plt.subplots()

ax.plot(x, x**2, label=r"$y = \alpha^2$")
ax.plot(x, x**3, label=r"$y = \alpha^3$")
ax.legend(loc=2) # upper left corner
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$y$')
ax.set_title('title');


# Or, alternatively, we can request that matplotlib uses LaTeX to render the text elements in the figure:

# In[30]:


# matplotlib.rcParams.update({'font.size': 18, 'text.usetex': True})


# In[31]:


fig, ax = plt.subplots()
ax.plot(x, x**2, label=r"$y = \alpha^2$")
ax.plot(x, x**3, label=r"$y = \alpha^3$")
ax.legend(loc=2) # upper left corner
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$y$')
ax.set_title('title');


# In[32]:


# restore
matplotlib.rcParams.update({'font.size': 12, 'font.family': 'sans', 'text.usetex': False})


# ### Setting colors, linewidths, linetypes

# #### Colors

# With matplotlib, we can define the colors of lines and other graphical elements in a number of ways. First of all, we can use the MATLAB-like syntax where `'b'` means blue, `'g'` means green, etc. The MATLAB API for selecting line styles are also supported: where, for example, 'b.-' means a blue line with dots:

# In[33]:


# MATLAB style line color and style 
ax.plot(x, x**2, 'b.-') # blue line with dots
ax.plot(x, x**3, 'g--') # green dashed line


# We can also define colors by their names or RGB hex codes and optionally provide an alpha value using the `color` and `alpha` keyword arguments:

# In[34]:


fig, ax = plt.subplots()

ax.plot(x, x+1, color="red", alpha=0.5) # half-transparant red
ax.plot(x, x+2, color="#1155dd")        # RGB hex code for a bluish color
ax.plot(x, x+3, color="#15cc55")        # RGB hex code for a greenish color


# #### Line and marker styles

# To change the line width, we can use the `linewidth` or `lw` keyword argument. The line style can be selected using the `linestyle` or `ls` keyword arguments:

# In[35]:


fig, ax = plt.subplots(figsize=(12,6))

ax.plot(x, x+1, color="blue", linewidth=0.25)
ax.plot(x, x+2, color="blue", linewidth=0.50)
ax.plot(x, x+3, color="blue", linewidth=1.00)
ax.plot(x, x+4, color="blue", linewidth=2.00)

# possible linestype options ‘-‘, ‘--’, ‘-.’, ‘:’, ‘steps’
ax.plot(x, x+5, color="red", lw=2, linestyle='-')
ax.plot(x, x+6, color="red", lw=2, ls='-.')
ax.plot(x, x+7, color="red", lw=2, ls=':')

# custom dash
line, = ax.plot(x, x+8, color="black", lw=1.50)
line.set_dashes([5, 10, 15, 10]) # format: line length, space length, ...

# possible marker symbols: marker = '+', 'o', '*', 's', ',', '.', '1', '2', '3', '4', ...
ax.plot(x, x+ 9, color="green", lw=2, ls='--', marker='+')
ax.plot(x, x+10, color="green", lw=2, ls='--', marker='o')
ax.plot(x, x+11, color="green", lw=2, ls='--', marker='s')
ax.plot(x, x+12, color="green", lw=2, ls='--', marker='1')

# marker size and color
ax.plot(x, x+13, color="purple", lw=1, ls='-', marker='o', markersize=2)
ax.plot(x, x+14, color="purple", lw=1, ls='-', marker='o', markersize=4)
ax.plot(x, x+15, color="purple", lw=1, ls='-', marker='o', markersize=8, markerfacecolor="red")
ax.plot(x, x+16, color="purple", lw=1, ls='-', marker='s', markersize=8, 
        markerfacecolor="yellow", markeredgewidth=2, markeredgecolor="blue");


# ### Control over axis appearance

# The appearance of the axes is an important aspect of a figure that we often need to modify to make a publication quality graphics. We need to be able to control where the ticks and labels are placed, modify the font size and possibly the labels used on the axes. In this section we will look at controling those properties in a matplotlib figure.

# #### Plot range

# The first thing we might want to configure is the ranges of the axes. We can do this using the `set_ylim` and `set_xlim` methods in the axis object, or `axis('tight')` for automatrically getting "tightly fitted" axes ranges:

# In[36]:


fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].plot(x, x**2, x, x**3)
axes[0].set_title("default axes ranges")

axes[1].plot(x, x**2, x, x**3)
axes[1].axis('tight')
axes[1].set_title("tight axes")

axes[2].plot(x, x**2, x, x**3)
axes[2].set_ylim([0, 60])
axes[2].set_xlim([2, 5])
axes[2].set_title("custom axes range");


# #### Logarithmic scale

# It is also possible to set a logarithmic scale for one or both axes. This functionality is in fact only one application of a more general transformation system in Matplotlib. Each of the axes' scales are set seperately using `set_xscale` and `set_yscale` methods which accept one parameter (with the value "log" in this case):

# In[37]:


fig, axes = plt.subplots(1, 2, figsize=(10,4))
      
axes[0].plot(x, x**2, x, np.exp(x))
axes[0].set_title("Normal scale")

axes[1].plot(x, x**2, x, np.exp(x))
axes[1].set_yscale("log")
axes[1].set_title("Logarithmic scale (y)");


# ### Placement of ticks and custom tick labels

# We can explicitly determine where we want the axis ticks with `set_xticks` and `set_yticks`, which both take a list of values for where on the axis the ticks are to be placed. We can also use the `set_xticklabels` and `set_yticklabels` methods to provide a list of custom text labels for each tick location:

# In[38]:


fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(x, x**2, x, x**3, lw=2)

ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xticklabels([r'$\alpha$', r'$\beta$', r'$\gamma$', r'$\delta$', r'$\epsilon$'], fontsize=18)

yticks = [0, 50, 100, 150]
ax.set_yticks(yticks)
ax.set_yticklabels(["$%.1f$" % y for y in yticks], fontsize=18); # use LaTeX formatted labels


# There are a number of more advanced methods for controlling major and minor tick placement in matplotlib figures, such as automatic placement according to different policies. See http://matplotlib.org/api/ticker_api.html for details.

# #### Scientific notation

# With large numbers on axes, it is often better use scientific notation:

# In[39]:


fig, ax = plt.subplots(1, 1)
      
ax.plot(x, x**2, x, np.exp(x))
ax.set_title("scientific notation")

ax.set_yticks([0, 50, 100, 150])

from matplotlib import ticker
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 
ax.yaxis.set_major_formatter(formatter) 


# ### Axis number and axis label spacing

# In[40]:


# distance between x and y axis and the numbers on the axes
matplotlib.rcParams['xtick.major.pad'] = 5
matplotlib.rcParams['ytick.major.pad'] = 5

fig, ax = plt.subplots(1, 1)
      
ax.plot(x, x**2, x, np.exp(x))
ax.set_yticks([0, 50, 100, 150])

ax.set_title("label and axis spacing")

# padding between axis label and axis numbers
ax.xaxis.labelpad = 5
ax.yaxis.labelpad = 5

ax.set_xlabel("x")
ax.set_ylabel("y");


# In[41]:


# restore defaults
matplotlib.rcParams['xtick.major.pad'] = 3
matplotlib.rcParams['ytick.major.pad'] = 3


# #### Axis position adjustments

# Unfortunately, when saving figures the labels are sometimes clipped, and it can be necessary to adjust the positions of axes a little bit. This can be done using `subplots_adjust`:

# In[42]:


fig, ax = plt.subplots(1, 1)
      
ax.plot(x, x**2, x, np.exp(x))
ax.set_yticks([0, 50, 100, 150])

ax.set_title("title")
ax.set_xlabel("x")
ax.set_ylabel("y")

fig.subplots_adjust(left=0.15, right=.9, bottom=0.1, top=0.9);


# ### Axis grid

# With the `grid` method in the axis object, we can turn on and off grid lines. We can also customize the appearance of the grid lines using the same keyword arguments as the `plot` function:

# In[43]:


fig, axes = plt.subplots(1, 2, figsize=(10,3))

# default grid appearance
axes[0].plot(x, x**2, x, x**3, lw=2)
axes[0].grid(True)

# custom grid appearance
axes[1].plot(x, x**2, x, x**3, lw=2)
axes[1].grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)


# ### Axis spines

# We can also change the properties of axis spines:

# In[44]:


fig, ax = plt.subplots(figsize=(6,2))

ax.spines['bottom'].set_color('blue')
ax.spines['top'].set_color('blue')

ax.spines['left'].set_color('red')
ax.spines['left'].set_linewidth(2)

# turn off axis spine to the right
ax.spines['right'].set_color("none")
ax.yaxis.tick_left() # only ticks on the left side


# ### Twin axes

# Sometimes it is useful to have dual x or y axes in a figure; for example, when plotting curves with different units together. Matplotlib supports this with the `twinx` and `twiny` functions:

# In[45]:


fig, ax1 = plt.subplots()

ax1.plot(x, x**2, lw=2, color="blue")
ax1.set_ylabel(r"area $(m^2)$", fontsize=18, color="blue")
for label in ax1.get_yticklabels():
    label.set_color("blue")
    
ax2 = ax1.twinx()
ax2.plot(x, x**3, lw=2, color="red")
ax2.set_ylabel(r"volume $(m^3)$", fontsize=18, color="red")
for label in ax2.get_yticklabels():
    label.set_color("red")


# ### Axes where x and y is zero

# In[46]:


fig, ax = plt.subplots()

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0)) # set position of x spine to x=0

ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))   # set position of y spine to y=0

xx = np.linspace(-0.75, 1., 100)
ax.plot(xx, xx**3);


# ## Other 2D plot styles

# In addition to the regular `plot` method, there are a number of other functions for generating different kind of plots. 
# * See the matplotlib plot gallery for a complete list of available plot types: http://matplotlib.org/gallery.html. 
# 
# Some of the more useful ones are show below:

# In[47]:


n = np.array([0,1,2,3,4,5])


# In[48]:


### Scatter plot
fig, axes = plt.subplots(1, 4, figsize=(12,3))
xx=n
axes[0].scatter(xx, xx + 0.25*np.random.randn(len(xx)))
axes[0].set_title("scatter")


# In[49]:


fig, axes = plt.subplots(1, 4, figsize=(12,3))

axes[0].scatter(xx, xx + 0.25*np.random.randn(len(xx)))
axes[0].set_title("scatter")

axes[1].step(n, n**2, lw=2)
axes[1].set_title("step")

axes[2].bar(n, n**2, align="center", width=0.5, alpha=0.5)
axes[2].set_title("bar")

axes[3].fill_between(x, x**2, x**3, color="green", alpha=0.5);
axes[3].set_title("fill_between");


# In[50]:


# polar plot using add_axes and polar projection
fig = plt.figure()
ax = fig.add_axes([0.0, 0.0, .6, .6], polar=True)
t = np.linspace(0, 2 * np.pi, 100)
ax.plot(t, t, color='blue', lw=3);


# In[51]:


# A histogram
n = np.random.randn(100000)
fig, axes = plt.subplots(1, 2, figsize=(12,4))

axes[0].hist(n)
axes[0].set_title("Default histogram")
axes[0].set_xlim((min(n), max(n)))

axes[1].hist(n, cumulative=True, bins=50)
axes[1].set_title("Cumulative detailed histogram")
axes[1].set_xlim((min(n), max(n)));


# ### Text annotation

# Annotating text in matplotlib figures can be done using the `text` function. It supports LaTeX formatting just like axis label texts and titles:

# In[52]:


fig, ax = plt.subplots()

ax.plot(xx, xx**2, xx, xx**3)

ax.text(0.15, 0.2, r"$y=x^2$", fontsize=20, color="blue")
ax.text(0.65, 0.1, r"$y=x^3$", fontsize=20, color="green");


# ### Figures with multiple subplots and insets

# Axes can be added to a matplotlib Figure canvas manually using `fig.add_axes` or using a sub-figure layout manager such as `subplots`, `subplot2grid`, or `gridspec`:

# #### subplots

# In[53]:


fig, ax = plt.subplots(2, 3)
fig.tight_layout()


# #### subplot2grid

# In[54]:


fig = plt.figure()
ax1 = plt.subplot2grid((3,3), (0,0), colspan=3)
ax2 = plt.subplot2grid((3,3), (1,0), colspan=2)
ax3 = plt.subplot2grid((3,3), (1,2), rowspan=2)
ax4 = plt.subplot2grid((3,3), (2,0))
ax5 = plt.subplot2grid((3,3), (2,1))
fig.tight_layout()


# #### gridspec

# In[55]:


import matplotlib.gridspec as gridspec


# In[56]:


fig = plt.figure()

gs = gridspec.GridSpec(2, 3, height_ratios=[2,1], width_ratios=[1,2,1])
for g in gs:
    ax = fig.add_subplot(g)
    
fig.tight_layout()


# #### add_axes

# Manually adding axes with `add_axes` is useful for adding insets to figures:

# In[57]:


fig, ax = plt.subplots()

ax.plot(xx, xx**2, xx, xx**3)
fig.tight_layout()

# inset
inset_ax = fig.add_axes([0.2, 0.55, 0.35, 0.35]) # X, Y, width, height

inset_ax.plot(xx, xx**2, xx, xx**3)
inset_ax.set_title('zoom near origin')

# set axis range
inset_ax.set_xlim(-.2, .2)
inset_ax.set_ylim(-.005, .01)

# set axis tick locations
inset_ax.set_yticks([0, 0.005, 0.01])
inset_ax.set_xticks([-0.1,0,.1]);


# ### Colormap and contour figures

# Colormaps and contour figures are useful for plotting functions of two variables. In most of these functions we will use a colormap to encode one dimension of the data. There are a number of predefined colormaps. It is relatively straightforward to define custom colormaps. For a list of pre-defined colormaps, see: http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps

# In[58]:


alpha = 0.7
phi_ext = 2 * np.pi * 0.5

def flux_qubit_potential(phi_m, phi_p):
    return 2 + alpha - 2 * np.cos(phi_p) * np.cos(phi_m) - alpha * np.cos(phi_ext - 2*phi_p)


# In[59]:


phi_m = np.linspace(0, 2*np.pi, 100)
phi_p = np.linspace(0, 2*np.pi, 100)
X,Y = np.meshgrid(phi_p, phi_m)
Z = flux_qubit_potential(X, Y).T


# #### pcolor

# In[60]:


fig, ax = plt.subplots()

p = ax.pcolor(X/(2*np.pi), Y/(2*np.pi), Z, cmap=matplotlib.cm.RdBu, vmin=abs(Z).min(), vmax=abs(Z).max())
cb = fig.colorbar(p, ax=ax)


# #### imshow

# In[61]:


fig, ax = plt.subplots()

im = ax.imshow(Z, cmap=matplotlib.cm.RdBu, vmin=abs(Z).min(), vmax=abs(Z).max(), extent=[0, 1, 0, 1])
im.set_interpolation('bilinear')

cb = fig.colorbar(im, ax=ax)


# #### contour

# In[62]:


fig, ax = plt.subplots()

cnt = ax.contour(Z, cmap=matplotlib.cm.RdBu, vmin=abs(Z).min(), vmax=abs(Z).max(), extent=[0, 1, 0, 1])


# ## 3D figures

# To use 3D graphics in matplotlib, we first need to create an instance of the `Axes3D` class. 3D axes can be added to a matplotlib figure canvas in exactly the same way as 2D axes; or, more conveniently, by passing a `projection='3d'` keyword argument to the `add_axes` or `add_subplot` methods.

# In[63]:


from mpl_toolkits.mplot3d.axes3d import Axes3D


# #### Surface plots

# In[64]:


fig = plt.figure(figsize=(14,6))

# `ax` is a 3D-aware axis instance because of the projection='3d' keyword argument to add_subplot
ax = fig.add_subplot(1, 2, 1, projection='3d')

p = ax.plot_surface(X, Y, Z, rstride=4, cstride=4, linewidth=0)

# surface_plot with color grading and color bar
ax = fig.add_subplot(1, 2, 2, projection='3d')
p = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, linewidth=0, antialiased=False)
cb = fig.colorbar(p, shrink=0.5)


# #### Wire-frame plot

# In[65]:


fig = plt.figure(figsize=(8,6))

ax = fig.add_subplot(1, 1, 1, projection='3d')

p = ax.plot_wireframe(X, Y, Z, rstride=4, cstride=4)


# #### Coutour plots with projections

# In[66]:


fig = plt.figure(figsize=(8,6))

ax = fig.add_subplot(1,1,1, projection='3d')

ax.plot_surface(X, Y, Z, rstride=4, cstride=4, alpha=0.25)
cset = ax.contour(X, Y, Z, zdir='z', offset=-np.pi, cmap=matplotlib.cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='x', offset=-np.pi, cmap=matplotlib.cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='y', offset=3*np.pi, cmap=matplotlib.cm.coolwarm)

ax.set_xlim3d(-np.pi, 2*np.pi);
ax.set_ylim3d(0, 3*np.pi);
ax.set_zlim3d(-np.pi, 2*np.pi);


# #### Change the view angle

# We can change the perspective of a 3D plot using the `view_init` method, which takes two arguments: `elevation` and `azimuth` angle (in degrees):

# In[ ]:


fig = plt.figure(figsize=(12,6))

ax = fig.add_subplot(1,2,1, projection='3d')
ax.plot_surface(X, Y, Z, rstride=4, cstride=4, alpha=0.25)
ax.view_init(30, 45)

ax = fig.add_subplot(1,2,2, projection='3d')
ax.plot_surface(X, Y, Z, rstride=4, cstride=4, alpha=0.25)
ax.view_init(70, 30)

fig.tight_layout()


# ### Animations

# Matplotlib also includes a simple API for generating animations for sequences of figures. With the `FuncAnimation` function we can generate a movie file from sequences of figures. The function takes the following arguments: `fig`, a figure canvas, `func`, a function that we provide which updates the figure, `init_func`, a function we provide to setup the figure, `frame`, the number of frames to generate, and `blit`, which tells the animation function to only update parts of the frame which have changed (for smoother animations):
# 
#     def init():
#         # setup figure
# 
#     def update(frame_counter):
#         # update figure for new frame
# 
#     anim = animation.FuncAnimation(fig, update, init_func=init, frames=200, blit=True)
# 
#     anim.save('animation.mp4', fps=30) # fps = frames per second
# 
# To use the animation features in matplotlib we first need to import the module `matplotlib.animation`:

# In[67]:


from matplotlib import animation


# In[63]:


# solve the ode problem of the double compound pendulum again

from scipy.integrate import odeint
from numpy import cos, sin

g = 9.82; L = 0.5; m = 0.1

def dx(x, t):
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    
    dx1 = 6.0/(m*L**2) * (2 * x3 - 3 * cos(x1-x2) * x4)/(16 - 9 * cos(x1-x2)**2)
    dx2 = 6.0/(m*L**2) * (8 * x4 - 3 * cos(x1-x2) * x3)/(16 - 9 * cos(x1-x2)**2)
    dx3 = -0.5 * m * L**2 * ( dx1 * dx2 * sin(x1-x2) + 3 * (g/L) * sin(x1))
    dx4 = -0.5 * m * L**2 * (-dx1 * dx2 * sin(x1-x2) + (g/L) * sin(x2))
    return [dx1, dx2, dx3, dx4]

x0 = [np.pi/2, np.pi/2, 0, 0]  # initial state
t = np.linspace(0, 10, 250) # time coordinates
x = odeint(dx, x0, t)    # solve the ODE problem


# Generate an animation that shows the positions of the pendulums as a function of time:

# In[ ]:


fig, ax = plt.subplots(figsize=(5,5))

ax.set_ylim([-1.5, 0.5])
ax.set_xlim([1, -1])

pendulum1, = ax.plot([], [], color="red", lw=2)
pendulum2, = ax.plot([], [], color="blue", lw=2)

def init():
    pendulum1.set_data([], [])
    pendulum2.set_data([], [])

def update(n): 
    # n = frame counter
    # calculate the positions of the pendulums
    x1 = + L * sin(x[n, 0])
    y1 = - L * cos(x[n, 0])
    x2 = x1 + L * sin(x[n, 1])
    y2 = y1 - L * cos(x[n, 1])
    
    # update the line data
    pendulum1.set_data([0 ,x1], [0 ,y1])
    pendulum2.set_data([x1,x2], [y1,y2])

anim = animation.FuncAnimation(fig, update, init_func=init, frames=len(t), blit=True)

# anim.save can be called in a few different ways, some which might or might not work
# on different platforms and with different versions of matplotlib and video encoders
#anim.save('animation.mp4', fps=20, extra_args=['-vcodec', 'libx264'], writer=animation.FFMpegWriter())
#anim.save('animation.mp4', fps=20, extra_args=['-vcodec', 'libx264'])
#anim.save('animation.mp4', fps=20, writer="ffmpeg", codec="libx264")
#anim.save('animation.mp4', fps=20, writer="avconv", codec="libx264")

plt.close(fig)


# Note: To generate the movie file we need to have either `ffmpeg` or `avconv` installed. Install it on Ubuntu using:
# 
#     $ sudo apt-get install ffmpeg
# 
# or (newer versions)
# 
#     $ sudo apt-get install libav-tools
# 
# On MacOSX, try: 
# 
#     $ sudo port install ffmpeg

# In[ ]:


from IPython.display import HTML
video = open("animation.mp4", "rb").read()
video_encoded = video.encode("base64")
video_tag = '<video controls alt="test" src="data:video/x-m4v;base64,{0}">'.format(video_encoded)
HTML(video_tag)


# ### Backends

# Matplotlib has a number of "backends" which are responsible for rendering graphs. The different backends are able to generate graphics with different formats and display/event loops. There is a distinction between noninteractive backends (such as 'agg', 'svg', 'pdf', etc.) that are only used to generate image files (e.g. with the `savefig` function), and interactive backends (such as Qt4Agg, GTK, MaxOSX) that can display a GUI window for interactively exploring figures. 
# 
# A list of available backends are:

# In[68]:


print(matplotlib.rcsetup.all_backends)


# The default backend, called `agg`, is based on a library for raster graphics which is great for generating raster formats like PNG.
# 
# Normally we don't need to bother with changing the default backend; but sometimes it can be useful to switch to, for example, PDF or GTKCairo (if you are using Linux) to produce high-quality vector graphics instead of raster based graphics. 

# #### Generating SVG with the svg backend

# In[70]:


#
# RESTART THE NOTEBOOK: the matplotlib backend can only be selected before pylab is imported!
# (e.g. Kernel > Restart)
# 
import matplotlib
matplotlib.use('svg')
import matplotlib.pylab as plt
import numpy
from IPython.display import Image, SVG


# In[71]:


#
# Now we are using the svg backend to produce SVG vector graphics
#
fig, ax = plt.subplots()
t = numpy.linspace(0, 10, 100)
ax.plot(t, numpy.cos(t)*numpy.sin(t))
plt.savefig("test.svg")


# In[72]:


#
# Show the produced SVG file. 
#
SVG(filename="test.svg")


# #### The IPython notebook inline backend

# When we use IPython notebook it is convenient to use a matplotlib backend that outputs the graphics embedded in the notebook file. To activate this backend, somewhere in the beginning on the notebook, we add:
# 
#     %matplotlib inline
# 
# It is also possible to activate inline matplotlib plotting with:
# 
#     %pylab inline
# 
# The difference is that `%pylab inline` imports a number of packages into the global address space (scipy, numpy), while `%matplotlib inline` only sets up inline plotting. In new notebooks created for IPython 1.0+, I would recommend using `%matplotlib inline`, since it is tidier and you have more control over which packages are imported and how. Commonly, scipy and numpy are imported separately with:
# 
#     import numpy as np
#     import scipy as sp
#     import matplotlib.pyplot as plt

# The inline backend has a number of configuration options that can be set by using the IPython magic command `%config` to update settings in `InlineBackend`. For example, we can switch to SVG figures or higher resolution figures with either:
# 
#     %config InlineBackend.figure_format='svg'
#      
# or:
# 
#     %config InlineBackend.figure_format='retina'
#     
# For more information, type:
# 
#     %config InlineBackend

# In[73]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='svg'")

import matplotlib.pylab as plt
import numpy


# In[74]:


#
# Now we are using the SVG vector graphics displaced inline in the notebook
#
fig, ax = plt.subplots()
t = numpy.linspace(0, 10, 100)
ax.plot(t, numpy.cos(t)*numpy.sin(t))
plt.savefig("test.svg")


# #### Interactive backend (this makes more sense in a python script file)

# In[75]:


#
# RESTART THE NOTEBOOK: the matplotlib backend can only be selected before pylab is imported!
# (e.g. Kernel > Restart)
# 
import matplotlib
matplotlib.use('Qt4Agg') # or for example MacOSX
import matplotlib.pylab as plt
import numpy as np


# In[76]:


# Now, open an interactive plot window with the Qt4Agg backend
fig, ax = plt.subplots()
t = np.linspace(0, 10, 100)
ax.plot(t, np.cos(t) * np.sin(t))
plt.show()


# Note that when we use an interactive backend, we must call `plt.show()` to make the figure appear on the screen.

# # Other visualization packages
# 
# In Python, there are multiple options for visualizing data. It can be challenging to decide one out of many for individual’s need.  Some of them are wrapper codes around matplotlib.
# The popular ones are:
# 
# * [Pandas](http://pandas.pydata.org/pandas-docs/stable/visualization.html):  
#     * Wrapper around matplotlib. Good for dataframe visuzalization
#     * See: http://pandas.pydata.org/pandas-docs/stable/visualization.html
# * [Seaborn](https://stanford.edu/~mwaskom/software/seaborn/)
#     * Wrapper around matplotlib
# * [Ggplot](http://ggplot.yhathq.com/)
# * [Bokeh](https://stanford.edu/~mwaskom/software/seaborn/)
#     * Not based on matplotlib. Similar to D3.js. 
#     * Mimic D3.js functionality
#     * Minimal coding compared to D3.js
# * [Pygal]()
#     * Can render images in SVG
# * Plotly(https://plot.ly/feed/) : Online plotting library
# 

# ## Further reading

# * http://www.matplotlib.org - The project web page for matplotlib.
# * https://github.com/matplotlib/matplotlib - The source code for matplotlib.
# * http://matplotlib.org/gallery.html - A large gallery showcaseing various types of plots matplotlib can create. Highly recommended! 
# * http://www.loria.fr/~rougier/teaching/matplotlib - A good matplotlib tutorial.
# * http://scipy-lectures.github.io/matplotlib/matplotlib.html - Another good matplotlib reference.
# 
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 19:58:54 2023

@author: Pooja
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 17:22:25 2022

@author: Pooja
DAY-10
"""

#------drinks data----------------------------------
#import required libraries
import pandas as pd
import numpy as np
#import the dataset of drinks
data=pd.read_csv('drinks.csv')
data.shape
data.head()
col=list(data.columns)
data.describe()
data.index
#which continent drinks more beer on average
data.groupby(['continent'])['beer_servings'].mean().idxmax()
#for each continent print the statistics for wine consumption
data.groupby('continent').wine_servings.describe()
#Print the mean alcoohol consumption per continent for every column
x=data.groupby('continent').mean()
#Print the median alcoohol consumption per continent for every column
data.groupby('continent').median()
#Print the mean, min and max values for 
#spirit consumption(output a dataframe).
#Print the mean, min and max values for spirit consumption(output a dataframe).
agg_data=data.groupby('continent').spirit_servings.agg(['mean', 'min', 'max'])
data.spirit_servings.hist(bins=6)
#-------------------------------------------
"""
Use pandas to read the file (titanics).
"""
import pandas as pd
D1=pd.read_csv('TST.csv')
"""
Use groupby method to calculate
the proportion of passengers that
survived by gender.
"""
D1.groupby('Sex').Survived.sum()
"""
Use groupby method to calculate
the proportion of passengers that died by gender.
"""
len(D1[D1['Survived']==0][D1['Sex']=='male'])
len(D1[D1['Survived']==0][D1['Sex']=='female'])
"""
Calculate the same proportion(survived) but by 
'Pclass and gender'.
"""
D1.groupby(['Sex', 'Pclass']).Survived.mean()
"""
create age categories:-
  childern(under 14yrs)
  adolescents(14-20yrs)
  adults(21-64 yrs)
  senior(65+ yrs)
and calculate survival proportion
by age category, Pclass and gender.
"""
data1=pd.read_csv("train_titanic.csv")
data1.Age.isna().sum()
Age=data1.Age.dropna()
Bins = [0, 14, 20, 64, 80]
BinLabels = ['under14','adolescents', 'adult','senior']

pd.cut(data1['Age'],Bins,labels=BinLabels)

#checking if there 
#are any nan values
np.isnan(D1['Age'])
#filling nana values
age=D1['Age'].fillna(int(D1['Age'].mean()), inplace=True)
pd.cut(D1['Age'],Bins,labels=BinLabels)
np.isnan(age)

#----------------------------------
#Import the necessary libraries
import numpy as np
import pandas as pd
#Import the dataset and assign to a variable
data2=pd.read_csv('student-mat.csv'),  sep=';')
#Display top 5 rows of data
data1.head()
data1.shape
col2=list(data2.columns)
#For the purpose of this exercise slice the dataframe from 'school' until the 'guardian' column
stud_alcoh = data2.iloc[: , :12]
stud_alcoh=data2.loc[:,'school':'guardian']
data2.columns
data1.info()
stud_alcoh.head()
#Create a lambda function that captalize strings.
c = lambda x: x.upper()
#Capitalize both Mjob and Fjob
stud_alcoh['Mjob'].apply(c)
stud_alcoh['Fjob'].apply(c)
#Print the last elements of the data set.
stud_alcoh.tail
#Did you notice the original dataframe is still lowercase? Why is that?
# Fix it and captalize Mjob and Fjob.
stud_alcoh['Mjob'] = stud_alcoh['Mjob'].apply(c)
stud_alcoh['Fjob'] = stud_alcoh['Fjob'].apply(c)
stud_alcoh.tail()


stud_alcoh['newcol']=np.arange(0,395)
del(stud_alcoh['newcol'])
"""
Create a function called majority that
 return 
a boolean value to a new 
column called legal_drinker
(Consider majority as older than 
 17 years old)
"""
def majority(x):
    if x > 17:
        return True
    else:
        return False
stud_alcoh['legal_drinker'] = stud_alcoh['age'].apply(majority)
#try to do this with .cut method!
stud_alcoh['legal_drinker']=pd.cut(stud_alcoh['age'], [0, 17,22], labels=[False, True])
stud_alcoh.head()
"""
Multiply every number of the dataset by 10.
"""
def times10(x):
    if type(x) is int:
        return 10 * x
    return x

stud_alcoh.applymap(times10).head(10)
stud_alcoh.set_index(stud_alcoh['famsize'])
#joining, merging dataframes------------------
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3'],
                        'C': ['C0', 'C1', 'C2', 'C3'],
                        'D': ['D0', 'D1', 'D2', 'D3']},
                        index=[0, 1, 2, 3])

df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                        'B': ['B4', 'B5', 'B6', 'B7'],
                        'C': ['C4', 'C5', 'C6', 'C7'],
                        'D': ['D4', 'D5', 'D6', 'D7']},
                         index=[4, 5, 6, 7]) 


df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                        'B': ['B8', 'B9', 'B10', 'B11'],
                        'C': ['C8', 'C9', 'C10', 'C11'],
                        'D': ['D8', 'D9', 'D10', 'D11']},
                        index=[8, 9, 10, 7])
# ## Concatenation
# 
# Concatenation basically glues together DataFrames. Keep in mind that dimensions should match along the axis you are concatenating on. You can use **pd.concat** and pass in a list of DataFrames to concatenate together:

pd.concat([df1,df2])
pd.concat([df1,df2,df3])


pd.concat([df1,df2,df3],axis=1)

left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})
   
right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K4'],
                          'C': ['C0', 'C1', 'C2', 'C3'],
                          'D': ['D0', 'D1', 'D2', 'D3']})    

# ## Merging
# 
# The **merge** function allows you to merge DataFrames together using a similar logic as merging SQL Tables together. For example:
pd.merge(left,right,how='outer',on='key')

left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                     'key2': ['K0', 'K1', 'K0', 'K1'],
                        'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3']})
    
right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                               'key2': ['K0', 'K0', 'K0', 'K0'],
                                  'C': ['C0', 'C1', 'C2', 'C3'],
                                  'D': ['D0', 'D1', 'D2', 'D3']})
pd.merge(left, right, on=['key1', 'key2'])
pd.merge(left, right, how='outer', on=['key1', 'key2'])
pd.merge(left, right, how='right', on=['key1', 'key2'])
pd.merge(left, right, how='left', on=['key1', 'key2'])
# ## Joining
# Joining is a convenient method for combining the columns of two potentially differently-indexed DataFrames into a single result DataFrame.
left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']},
                      index=['K0', 'K1', 'K2']) 
right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                    'D': ['D0', 'D2', 'D3']},
                      index=['K0', 'K2', 'K3'])
left.join(right)
left.join(right, how='outer')
#---------------------------------------------------
#data visualization
#---------------------------------------------------
"""
The basic steps to creating 
plots with matplotlib are:              
1 Prepare data     
2 Create plot     
3 Plot     
4 Customize plot    
5 Save plot    
6 Show plot
"""
import matplotlib.pyplot as plt
import numpy as np
#genrate 1D data
x = np.linspace(0, 10, 100)
y = np.cos(x) 
z = np.sin(x)
plt.scatter(x[:20],y[:20], color='r',linewidth=2.5, linestyle='--', facecolor='yellow', or)
plt.xlabel("value of x")
plt.ylabel("this is y")
plt.ylim(-1.5,1.5)
plt.title("This is the first plot")
plt.show()
#genrate 2D data
data = 2 * np.random.random((10, 10)) 
data2 = 3 * np.random.random((10, 10)) 
#from matplotlib.cbook import get_sample_data 
#img = np.load(get_sample_data('axes_grid/bivariate_normal.npy'))
#im = ax.imshow(img,cmap='gist_earth',interpolation='nearest',vmin=-2,vmax=2)
gdp_cap=[]
for i in range(50):
    n=np.random.randint(155, 389)
    n=n/100
    gdp_cap.append(n)
life_exp = []
for i in range(50):
    n=np.random.randint(100, 150)
    n=n/100
    life_exp.append(n)
pop = []
for i in range(50):
    n=np.random.randint(190, 230)
    n=n/100
    pop.append(n)       
import matplotlib.pyplot as plt
plt.plot(x,z)
"""
histogram-----------------------------------
"""
plt.hist(y)
plt.hist(pop)
plt.hist(pop, bins= 5, color='red')
plt.cla()               #Clear an axis >>> 
plt.clf()               #Clear the entire figure >>> 
plt.close()  

plt.xlabel('my data')
plt.ylabel('your data')
plt.xticks([0, 1, 2,2.3])
plt.yticks([0,2,4,8])           #Close a window
plt.hist(pop, color='yellow')
#plt.show()
plt.savefig('plot.png', dpi=300)
plt.savefig('hist2.png', transparent=True)   
plt.xscale('log')
plt.yscale('log')
plt.hist(pop, orientation='horizontal')# orientation changes the look of plot try it with 'vertical'

"""
Scatter ------------------------------------------------------
"""
"""
import numpy as np
x= np.random.rand(5)
y=np.random.rand(5)
s1= [10,20,80,40,50]
c=[3,3,5,6,7]
c1=np.random.rand(5)
"""
plt.scatter(x,y)
plt.close()
plt.scatter(x[:5],y[:5], s=s1, c= c)
plt.scatter(gdp_cap, life_exp, s =pop)
plt.xscale('log') 
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
plt.xticks([1000, 10000, 100000],['1k', '10k', '100k'])
plt.legend(["curve1"])
from sklearn.datasets import load_iris
data = load_iris()
import pandas as pd
from sklearn import preprocessing
# load the iris dataset
iris = load_iris()
X = data.data
y=data.target
x=np.array(X)
data_x=pd.DataFrame(X)
pd.scatter_matrix(data_x)
#---------------------------
#pie chart
#---------------------
l = 'Python', 'C++', 'Ruby', 'Java'
sizes = [310, 30, 145, 110]
col = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
exp = (0, 0, 0, 0.2)  # explode 1st slice
plt.pie(sizes, labels=l, colors=col, explode=exp, startangle=50)
plt.pie(sizes, explode=exp, labels=l, colors=col,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.tight_layout()
plt.axis('equal')
plt.show()
#------------------
#bar plot
x = np.arange(4)
money = [1.5e5, 2.5e6, 5.5e6, 2.0e7]
a,b,c,d=plt.bar(x, money)
a.set_facecolor('r')
b.set_facecolor('g')
c.set_facecolor('b')
d.set_facecolor('black')
plt.xticks(x, ('Bill', 'Fred', 'Mary', 'Sue'))
plt.show()
plt.barh(x, money)
import pandas as pd
df2 = pd.DataFrame(np.random.rand(10, 4), 
                   columns=['a', 'b', 'c', 'd'])
df2.plot.bar()
df2.plot.bar(stacked=True)
"""
box plot:
    The box plot (a.k.a. box and whisker diagram) is a 
    standardized way of displaying the distribution of 
    data based on the five number summary: minimum, 
    first quartile, median, third quartile, and maximum.
    In the simplest box plot the central rectangle spans
    the first quartile to the third quartile 
    (the interquartile range or IQR). 
    A segment inside
    the rectangle shows the median and 
    "whiskers" 
    above and below the box show the locations of the 
    minimum and maximum.
If the data happens to be normally distributed,
IQR = 1.35 σ
where σ is the population standard deviation.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.DataFrame(np.random.rand(10, 5),
                  columns=['A', 'B', 'C', 'D', 'E'], index=['a','b','c','d','e','f','g','h','i','j'])
df.loc[3,'B']=6
color = dict(boxes='DarkGreen', 
             whiskers='DarkOrange', 
             medians='DarkBlue', caps='Gray')
df.plot.box(color=color, sym='r+')
plt.boxplot(df['B'])
s=np.random.rand(50)*100
c=np.ones(25)*50
fh=np.random.randn(10)*100+100
fl=np.random.randn(10)*-100
dd=np.concatenate((s,c,fh,fl),0)
dd=pd.DataFrame(dd)
plt.hist(dd)
plt.boxplot(dd)
dd.columns

df['A']
df.loc[df['C']>0.6]=np.nan
f.loc['d']

f.iloc[:4]







#--------------------------------
from pandas.tools.plotting import parallel_coordinates
from sklearn.datasets import load_iris
data=load_iris()
dataframe=pd.DataFrame(data.data,columns=data.feature_names)
dataframe['class']=data.target
dataframe.shape
plt.figure()
parallel_coordinates(dataframe[:10,::] , 'class')
dataframe.groupby('class').size()
print(dd.groupby('class').size())
plt.savefig('pp.png')
df.plot.area()
dataframe.plot.area()

#---------------------
#date-time module
d1 = "10/24/2017"
d2 = "11/24/2016"
max(d1,d2)
d1 - d2
import datetime
d1 = datetime.date(2016,11,24)
d2 = datetime.date(2017,10,24)
max(d1,d2)
print(d2 - d1)
century_start = datetime.date(2000,1,1)
today = datetime.date.today()
print(century_start,today)
print("We are",today-century_start,"days into this century")

century_start = datetime.datetime(2000,1,1,0,0,0)
time_now = datetime.datetime.now()


time_since_century_start = time_now - century_start
print("days since century start",
      time_since_century_start.days)
print("seconds since century start",
      time_since_century_start.total_seconds())
print("minutes since century start",
      time_since_century_start.total_seconds()/60)
print("hours since century start",
      time_since_century_start.total_seconds()/60/60)
dtn = datetime.datetime.now()
tn = dtn.time()
print(tn)
today=datetime.date.today()
fdl=today+datetime.timedelta(days=5, minutes=89, seconds=9)
print(fdl)
#--------------feature scalling
from sklearn.datasets import load_iris
data = load_iris()
import pandas as pd
from sklearn import preprocessing
iris = load_iris()
print(iris.data.shape)

# separate the data from the target attributes
X = iris.data
y = iris.target

normalized_X = pd.DataFrame(preprocessing.normalize(X), columns=iris.feature_names)
normalized_X['class']=iris.target
from pandas.tools.plotting import parallel_coordinates
parallel_coordinates(normalized_X,'class')
standardized_X = preprocessing.scale(X)

#!/usr/bin/env python
# coding: utf-8

# # Introduction to Pandas

# In[37]:


import pandas as pd


# We will open the data set on list of passenger on ill-fated Titanic cruise

# In[38]:


# Use CSV reader  
df = pd.read_csv("titanic.csv")


# See more about [read_csv](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html#pandas.read_csv) in pandas documentation

# See first few rows of the data frame

# In[39]:


df.head()


# In[40]:


# Display data type of the variable df
type(df)


# We can display data types of individual columns of the data read into data frame using *dtypes* 

# In[41]:


df.dtypes


# To find the size or the shape of dataframe object

# In[42]:


df.shape


# To get summarized information about data frame

# In[43]:


df.info()


# We can get statistical description of the data using *describe()* method of the dataframe

# In[44]:


df.describe()


# *describe()* is a part of descriptive statistics methods available to pandas object. See [documentation](http://pandas.pydata.org/pandas-docs/stable/api.html#computations-descriptive-stats) for different available functions.

# ## Referencing

# * Each column of the dataframe is referenced by its "Label".
# * Similar to numpy array we can use index based referencing to reference elements in each column of the data frame.

# In[45]:


df['Age'][0:10]


# df['Age'].mean()

# Each column of the dataframe is pandas object series. So all descriptive statistics methods are 
#available to each of the columns.

#
# In[46]:


# Compute median age
df['age'].median()


# Check if the above median ignores *NaN* 
# 
# Multiple columns can be referenced by passing a list of columns to dataframe object as shown below.

# In[47]:


MyColumns = ['sex', 'pclass','age']
df[MyColumns].head()


# ## Filtering

# Dataframe object can take logical statements as inputs. Depending upon value of this logical index, it will return the resulting dataframe object.

# In[48]:


# Select all passenger with age greater than 60

df_AgeMoreThan60 = df[df['age']>60]
df_AgeMoreThan60.head()


# In[49]:


# Select all passengers with age less than or equal to 15
df_AgeLessThan15=df[df['age']<=15]

# Number of passengers with Age less than or equal to 15
df_AgeLessThan15['age'].count()


# Passengers whose age is more than 60 and are male

# Lets see only passengers who are male and above 60 years old

# In[50]:


# Method-1: apply two filters sepeartly
df_AgeMoreThan60 = df[df['age'] > 60]
temp1 = df_AgeMoreThan60[df['sex']=='male']
temp1 ['sex'].head()


# In[51]:


# Method-2: Applying  filters together
SurvivedMaleMoreThan60 = df[(df.age>60) & (df.sex=='male') ]
SurvivedMaleMoreThan60['sex'].head()


# In[52]:


# Method-2: Applying two or more filters together
SurvivedMaleMoreThan60 = df[(df['age']>60) & (df['sex']=='male') & (df['survived']==1)]
SurvivedMaleMoreThan60['sex'].head()


# ## Tabulation

# In[53]:


mySeries = df['pclass']


# method *[value_counts()](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.value_counts.html)* will return counts of unique values

# In[54]:


mySeries.value_counts()
df['survived'].value_counts()


# In[55]:


#Cross tabulation
pd.crosstab(df['sex'],df['pclass'])
pd.crosstab(df['sex'],df['survived'])

# ## Dropping rows and columns

# In[56]:


# Drop columns
df.drop('age').head() 
df.drop('age',axis=1).head() 
# Note axis=1 indicates that label "Age" is along dimension (index) 1 (0 for rows, 1 for column)


# ## Data frame with row lables and column labels

# In[57]:


#Generate some data
import numpy as np
data = np.random.random(16)
data =  data.reshape(4,4)
data


# In[58]:


# Generate column and row labels
ColumnLables=['One','Two','Three','Four']
RowLables =['Ohio','Colarado','Utah','New York']


# In[59]:


# Use DataFrame method to create dataframe object
df2=pd.DataFrame(data,RowLables,ColumnLables)


# In[60]:


df2.drop('Utah')


# In[61]:


df4=df.dropna(axis=1)
df4.shape
df5=df.interpolate(method="linear")
df5.isna().sum()
# ## Combining, merging and concatenating two data frames

# We will create two dataframe objects

# In[62]:


df2=pd.DataFrame(data,RowLables,ColumnLables)
df3=pd.DataFrame(data*4,RowLables,ColumnLables)

df2.shape
df3.shape
# ### **Merge pandas objects **
# by performing a database-style join operation by columns or indexes.
# see [merge documentation](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.merge.html#pandas.merge) for details

# In[63]:


#merge
df4=pd.merge(df2,df3) # default inner join
df4


# In[64]:


df5=pd.merge(df2,df3,how='outer')
df5=pd.merge(df2,df3, how='right')
df5


# ### **Concatenate pandas objects** 
# along a particular axis with optional set logic along the other axes.
# see [concat documentation](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.concat.html#pandas.concat) for details

# In[65]:


pd.concat([df2,df2])


# ## Removing duplicates
# 
# *drop_duplicates* will drop duplicate rows

# In[66]:


df6=pd.concat([df2,df2])

df6.drop_duplicates()


# ## Discreatization and Binning

# ### Cut method
# *[cut](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html#pandas.cut)* method return indices of half-open bins to which each value of x belongs. See documentation details for different options.

# In[67]:


PassengerAge = df['age']
PassengerAge = PassengerAge.dropna()
Bins = [0, 10,15,30, 40, 60, 80]

pd.cut(PassengerAge,Bins).head()


# ### Cut with labels for generated bins
# 
# We can also apply "Labels" to each of the generated bin

# In[68]:


PassengerAge = df['Age']

PassengerAge = PassengerAge.dropna()

Bins = [0, 10,15,30, 40, 60, 80]

BinLabels = ['Toddler','Young', 'Adult','In Early 50s', 'In 60s', 'Gerand'] #labels for generated bins

pd.cut(PassengerAge,Bins,labels=BinLabels).head()


# ### Use of precision to cut numbers

# In[69]:


import numpy as np
data = np.random.rand(20)
pd.cut(data,4,precision=2)

