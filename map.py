import plotly.plotly as py
import plotly.graph_objs as go
import geopandas
import shapefile
import shapely
from plotly.figure_factory._county_choropleth import create_choropleth
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

fips = ['06021', '06023', '06027',
        '06029', '06033', '06059',
        '06047', '06049', '06051',
        '06055', '06061']
values = range(len(fips))

fig = create_choropleth(fips=fips, values=values)
iplot(fig)