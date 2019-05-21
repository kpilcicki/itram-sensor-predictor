import plotly.offline as py
import plotly.graph_objs as go

plotly_layout = go.Layout(
        xaxis=dict(
            title='t[s]',
            titlefont=dict(
                family='Arial, sans-serif',
                size=18,
                color='lightgrey'
            ),
            showticklabels=True,
            tickfont=dict(
                family='Old Standard TT, serif',
                size=14,
                color='black'
            ),
            exponentformat='e',
            showexponent='all'
        ),
        yaxis=dict(
            title='A[m]',
            titlefont=dict(
                family='Arial, sans-serif',
                size=18,
                color='lightgrey'
            ),
            showticklabels=True,
            tickfont=dict(
                family='Old Standard TT, serif',
                size=14,
                color='black'
            ),
            exponentformat='e',
            showexponent='all'
        )
    )

def plot(graph):
  data = [go.Scatter(
            x=graph['x'],
            y=graph['y'],
            mode="markers"
        )]

  figure = go.Figure(data=data, layout=plotly_layout)
  
  py.plot(figure, filename='graph')

def show_multiple_series(values):
    layout = plotly_layout
    data = []
    for correlation in values:
        data.append(go.Scatter(
            x=list(range(len(correlation))),
            y=correlation,
            mode="markers" if True else 'lines',
        ))

    figure = go.Figure(data=data, layout=layout)
    
    py.plot(figure, filename='mgraph')