---
layout: post
published: true
title: Where are the Starbucks stores located in India ?
date: '2017-11-22'
---
While sipping a cup of coffee, or just passing by a Starbucks store, how many of us actually think about the role analytics played in opening that store? Probably not many of us. But Analytics does play a major role in taking those business decisions.

These decisions are backed by many data points. But to take those decisions fast the representation of those data points matters a lot. 

In this analysis we can see that how simple it is to generate such a powerful visualisation where Starbucks located in India are plotted on Map(Red marker: store with more footfall, Green markers: store with less footfall) using folium package and on the top of that we can even generate an individual graph for every data point.
This analysis can be useful in analysing the performance of starbucks store and in deciding where to open new store.
This graph is currently centered around Mumbai region. Zoom out to see for other cities in India and click on store markers to view footfalls.

<iframe src = "http://www.shwetkmishra.com/starbucks_india.html" width = "100%" height = "1000">
         Sorry your browser does not support inline frames.
</iframe>

Here's the complete code snippet:

<script src="https://gist.github.com/shwetkm/6e138f6008b53ff3f19d9687086491ce.js"></script>

The original location data is taken from Kaggle and foot-fall data is generated randomly(dummy) just for the visualization purpose. Dataset used here can be found on my git <a href="https://github.com/shwetkm/Starbucks_India_Location_Analysis">repo</a>.

