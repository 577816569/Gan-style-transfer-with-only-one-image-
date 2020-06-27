# Some explorations on GAN with style transfer

This work is only a small attempt to explore the field of GAN(Generative Adversarial Networks). The result of this work is really bad due to my lack of understanding in the field of deep learning and the terrible training environment(I trained the network on Colab). So if anyone see this repo(probably no one) and find a lot of mistakes please ignore them. 




## About this work
There have been already a lot of work explored the possibility to use GAN achieve style transfer. For example:

Cycle GAN : [[Paper]](https://arxiv.org/abs/1703.10593)

Cartoon GAN : [[Paper]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf)

Gated GAN : [[Paper]](https://arxiv.org/abs/1904.02296)

These GAN could generate very good results. However, to train a GAN, we need thousands of paired/unpaired image datas, and if we want to achieve transfer only one image's style like the original paper wrote by Gatys [[Paper]](https://ieeexplore.ieee.org/document/7780634), that could be a problem. Also, Gatys' method of style transfer is online method, so the efficiency is not as high as GAN base method.

In this work, I use some simple tricks to achieve transfer only one image's style to another with GAN.

## Method

To solve the problem of lack of datasets, I cut the orignial style image into 256 square images randomly and the size of each image is 16 * 16. By conbining these small images, we can get a 256 * 256 style image. We can use this method to generate as many style images as we want. 

<img src="https://github.com/577816569/Some-explorations-on-gan-with-styletransfer/blob/master/style/mosaic.jpg" width = "300" height = "200" alt="Original style image" align=center /><img src="https://github.com/577816569/Some-explorations-on-gan-with-styletransfer/blob/master/style/0.jpg" width = "200" height = "200" alt="Transfered style image" align=center />

&emsp;&emsp;&emsp;&emsp;Original style image &emsp;&emsp;&emsp;                     Transfered style image
 
Also, the code of these work is base on the this 



## Usage

You can rename the current file by clicking the file name in the navigation bar or by clicking the **Rename** button in the file explorer.

## Delete a file

You can delete the current file by clicking the **Remove** button in the file explorer. The file will be moved into the **Trash** folder and automatically deleted after 7 days of inactivity.

## Export a file

You can export the current file by clicking **Export to disk** in the menu. You can choose to export the file as plain Markdown, as HTML using a Handlebars template or as a PDF.


# Synchronization

Synchronization is one of the biggest features of StackEdit. It enables you to synchronize any file in your workspace with other files stored in your **Google Drive**, your **Dropbox** and your **GitHub** accounts. This allows you to keep writing on other devices, collaborate with people you share the file with, integrate easily into your workflow... The synchronization mechanism takes place every minute in the background, downloading, merging, and uploading file modifications.

There are two types of synchronization and they can complement each other:

- The workspace synchronization will sync all your files, folders and settings automatically. This will allow you to fetch your workspace on any other device.
	> To start syncing your workspace, just sign in with Google in the menu.

- The file synchronization will keep one file of the workspace synced with one or multiple files in **Google Drive**, **Dropbox** or **GitHub**.
	> Before starting to sync files, you must link an account in the **Synchronize** sub-menu.

## Open a file

You can open a file from **Google Drive**, **Dropbox** or **GitHub** by opening the **Synchronize** sub-menu and clicking **Open from**. Once opened in the workspace, any modification in the file will be automatically synced.

## Save a file

You can save any file of the workspace to **Google Drive**, **Dropbox** or **GitHub** by opening the **Synchronize** sub-menu and clicking **Save on**. Even if a file in the workspace is already synced, you can save it to another location. StackEdit can sync one file with multiple locations and accounts.

## Synchronize a file

Once your file is linked to a synchronized location, StackEdit will periodically synchronize it by downloading/uploading any modification. A merge will be performed if necessary and conflicts will be resolved.

If you just have modified your file and you want to force syncing, click the **Synchronize now** button in the navigation bar.

> **Note:** The **Synchronize now** button is disabled if you have no file to synchronize.

## Manage file synchronization

Since one file can be synced with multiple locations, you can list and manage synchronized locations by clicking **File synchronization** in the **Synchronize** sub-menu. This allows you to list and remove synchronized locations that are linked to your file.


# Publication

Publishing in StackEdit makes it simple for you to publish online your files. Once you're happy with a file, you can publish it to different hosting platforms like **Blogger**, **Dropbox**, **Gist**, **GitHub**, **Google Drive**, **WordPress** and **Zendesk**. With [Handlebars templates](http://handlebarsjs.com/), you have full control over what you export.

> Before starting to publish, you must link an account in the **Publish** sub-menu.

## Publish a File

You can publish your file by opening the **Publish** sub-menu and by clicking **Publish to**. For some locations, you can choose between the following formats:

- Markdown: publish the Markdown text on a website that can interpret it (**GitHub** for instance),
- HTML: publish the file converted to HTML via a Handlebars template (on a blog for example).

## Update a publication

After publishing, StackEdit keeps your file linked to that publication which makes it easy for you to re-publish it. Once you have modified your file and you want to update your publication, click on the **Publish now** button in the navigation bar.

> **Note:** The **Publish now** button is disabled if your file has not been published yet.

## Manage file publication

Since one file can be published to multiple locations, you can list and manage publish locations by clicking **File publication** in the **Publish** sub-menu. This allows you to list and remove publication locations that are linked to your file.


# Markdown extensions

StackEdit extends the standard Markdown syntax by adding extra **Markdown extensions**, providing you with some nice features.

> **ProTip:** You can disable any **Markdown extension** in the **File properties** dialog.


## SmartyPants

SmartyPants converts ASCII punctuation characters into "smart" typographic punctuation HTML entities. For example:

|                |ASCII                          |HTML                         |
|----------------|-------------------------------|-----------------------------|
|Single backticks|`'Isn't this fun?'`            |'Isn't this fun?'            |
|Quotes          |`"Isn't this fun?"`            |"Isn't this fun?"            |
|Dashes          |`-- is en-dash, --- is em-dash`|-- is en-dash, --- is em-dash|


## KaTeX

You can render LaTeX mathematical expressions using [KaTeX](https://khan.github.io/KaTeX/):

The *Gamma function* satisfying $\Gamma(n) = (n-1)!\quad\forall n\in\mathbb N$ is via the Euler integral

$$
\Gamma(z) = \int_0^\infty t^{z-1}e^{-t}dt\,.
$$

> You can find more information about **LaTeX** mathematical expressions [here](http://meta.math.stackexchange.com/questions/5020/mathjax-basic-tutorial-and-quick-reference).


## UML diagrams

You can render UML diagrams using [Mermaid](https://mermaidjs.github.io/). For example, this will produce a sequence diagram:

```mermaid
sequenceDiagram
Alice ->> Bob: Hello Bob, how are you?
Bob-->>John: How about you John?
Bob--x Alice: I am good thanks!
Bob-x John: I am good thanks!
Note right of John: Bob thinks a long<br/>long time, so long<br/>that the text does<br/>not fit on a row.

Bob-->Alice: Checking with John...
Alice->John: Yes... John, how are you?
```

And this will produce a flow chart:

```mermaid
graph LR
A[Square Rect] -- Link text --> B((Circle))
A --> C(Round Rect)
B --> D{Rhombus}
C --> D
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTYzMzE0MzQyOSw4MTA3MDMyNzEsLTM4ND
QwNzA2NCwtMTY4OTgxNTQ4MiwxMjM5MTY0Nzk1LDE5ODE3NTgy
NjMsMjA4Njk2NzUzMywxODIzMjgyMzY4LC0xNjMwNjM1MTkxLC
00MTQ0MzkyMjIsODk1Njk5NDYyLC0xMTE5MDQxNzIxLC0xNzE5
MzM0NjQ5LC0xNDAyMjc4MzU1LDMxMTY3MTAsLTE0ODU4MzY3MD
YsLTE0NTQ1MjI3NSwxNjQ3MjIwNjZdfQ==
-->