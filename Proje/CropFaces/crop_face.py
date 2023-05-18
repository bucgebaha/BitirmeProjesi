import cv2
  
dataset_real_dir = 'C:\\Users\\bhabu\\Desktop\\Bitirme\\Proje\\Dataset\\Real\\real ('
dataset_cropped_real_dir = 'C:\\Users\\bhabu\\Desktop\\Bitirme\\Proje\\DatasetCropped\\train\\real\\'

dataset_fake_dir ='C:\\Users\\bhabu\\Desktop\\Bitirme\\Proje\\Dataset\\Fake\\fake ('
dataset_cropped_fake_dir ='C:\\Users\\bhabu\\Desktop\\Bitirme\\Proje\\DatasetCropped\\train\\fake\\'

for j in range(1,2677):
    # Read the input image
    img = cv2.imread(dataset_real_dir + str(j) + ').jpg')
      
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
    	cv2.rectangle(img, (x, y), (x+w, y+h),
    				(0, 0, 255), 2)
    	
    	faces = img[y:y + h, x:x + w]
    	cv2.imwrite(dataset_cropped_real_dir + str(j) + '.jpg', faces)
        
for j in range(1,2677):
    # Read the input image
    img = cv2.imread(dataset_fake_dir + str(j) + ').jpg')
      
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
    	cv2.rectangle(img, (x, y), (x+w, y+h),
    				(0, 0, 255), 2)
    	
    	faces = img[y:y + h, x:x + w]
    	cv2.imwrite(dataset_cropped_fake_dir + str(j) + '.jpg', faces)