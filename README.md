### wiki-tSNE

This IPython notebook will show you how to cluster and visualize a set of documents, articles, or texts as in [this demo](http://www.genekogan.com/works/wiki-tSNE). The included example clusters articles visited in the [WikiGame](http://thewikigame.com/). To do that, you must copy the table __LAST GAME RESULTS__ on the left of the website and save it as text in `snapshot.txt`. Then in the python file, enter the lines of the users whose link path you want to display.

The notebook derives a clustering by first converting a set of documents into a [tf-idf matrix](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) which is a representation of each document as a vector in which each element represents the relative importance of a unique term to that document. Using that representation, we can reduce its dimension to 2 using [t-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding), and then save it to a json file, along with the order of all the link paths.

The folder `visualize` contains a [p5.js](http://www.p5js.org) sketch which displays the results in a browser.