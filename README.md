# Labotarorio de Machine Learning  - ITBA 2017

# Proyecto Final  :  Clasificacion de Troqueles de Medicamentos en Prescripciones Medicas 

# Descripcion del Ambiente 

El objetivo del trabajo es el analisis de imagenes de prescripciones medicas (recetas) teniendo en cuenta los siguientes controles :  

	* Troqueles de medicamentos adheridos.
	* Troqueles de medicamentos defectuosos o con imperfecciones (tachaduras , cortados , ilegibles) .

Estos controles se realizan en el ambito de las auditorias de medicamentos como servicios a las obras sociales y prepagas en forma visual , que requiere de un operador de revise cada receta de modo manual y que en general observe segun su criterio algunos aspectos formales que las entidades de salud establecen en su Norma Operativa para el control de las prescripciones medicas .  
 
El objectivo del trabajo es reducir el tiempo de procesamiento a traves del analisis de estas imagenes escaneadas  y poder disminuir la cantidad de recetas visualizadas en forma manual  y clasificarlas a partir del umbral de error que proporcione el analisis de las mismas .


# Imagen de Ejemplo (Ver ./Image13112017.jpg , ./Image14112017.jpg) 


# Documentacion y links consultados para este trabajo 

http://www.diva-portal.org/smash/get/diva2:1164104/FULLTEXT01.pdf “Reading barcodes with neural networks” by  Fredrik Fridborn 
https://machinelearningmastery.com/image-augmentation-deep-learning-keras/
https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/
https://www.pyimagesearch.com/2016/08/08/k-nn-classifier-for-image-classification/
https://www.pyimagesearch.com/2016/06/20/detecting-cats-in-images-with-opencv/
https://www.pyimagesearch.com/2014/12/15/real-time-barcode-detection-video-python-opencv/
https://kukuruku.co/post/something-about-cats-dogs-machine-and-deep-learning/
https://www.pyimagesearch.com/2016/08/15/how-to-tune-hyperparameters-with-python-and-scikit-learn/
https://matplotlib.org/examples/color/colormaps_reference.html
https://github.com/yushulx/opencv-python-webcam-barcode-reader
https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
https://www.bonaccorso.eu/2016/08/06/cifar-10-image-classification-with-keras-convnet/
https://pdfs.semanticscholar.org/d017/8e391afef701163a977369c7339a0802aa7c.pdf
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2859730/  (A Bayesian Algorithm for Reading 1D Barcodes) 
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4022351/ ( Supervised DNA Barcodes species classification: analysis, comparisons and results ) 
http://dimacs.rutgers.edu/Workshops/BarcodeResearchChallenges2007/
https://opencv.org/
http://pdftoimage.com/es/ 
https://github.com/jrosebr1/imutils

# Datasets

Se dispone de imagenes de prescripciones medicas que fueron publicadas en distintos formatos, como por ejemplo documentos con extension docx, pdf, tif, jpg 

Se realizo un preprocesamiento para convertir esos documentos a imagenes con formato jpg para luego extraer los codigos de barras de los troqueles de medicamentos en las prescripciones en imagenes individuales. 

Se cuenta con un datasets de entrenamiento de 248 imagenes en formato .jpg de 430 x 430 en blanco y negro, divididas de la siguiente forma : 

	* 124 imagenes de troqueles de medicamentos consideradas correctas . 
	* 124 imagenes que se consideran incorrectas para la clasificacion que se pretende realizar .
        

Se cuenta con un dataset de testing de 80 imagenes con las mismas dimensiones mencionadas anteriormente. 

# Notebooks 

	* Detalle de las pruebas realizadas 
https://github.com/gebasilio/ML_TPFINAL_Troqueles_Medicos/0-ML_TPFINAL_Imagenes_Troqueles_Pruebas.ipynb

	* Preprocesamiento de Imagenes 
https://github.com/gebasilio/ML_TPFINAL_Troqueles_Medicos/1-ML_TPFINAL_Imagenes_Troqueles_PREPROCESO.ipynb

	* Modelos de Clasificacion CNN  
https://github.com/gebasilio/ML_TPFINAL_Troqueles_Medicos/2-ML_TPFINAL_Imagenes_Troqueles_MODELO_CNN.ipynb

	* Modelos de Clasificacion CLF  
https://github.com/gebasilio/ML_TPFINAL_Troqueles_Medicos/3-ML_TPFINAL_Imagenes_Troqueles_MODELO_CLF.ipynb 

	* Modelos de Clasificacion KNN 
https://github.com/gebasilio/ML_TPFINAL_Troqueles_Medicos/4-ML_TPFINAL_Imagenes_Troqueles_MODELO_KNN.ipynb

	* Modelos Pre-trained VGG16 
https://github.com/gebasilio/ML_TPFINAL_Troqueles_Medicos/5-ML_TPFINAL_Imagenes_Troqueles_MODELO_VGG16.ipynb


      

