import cv2

dataset_real_dir = 'C:\\Users\\bhabu\\Desktop\\Bitirme\\Proje\\Dataset\\Real\\'
real_videos_dir = 'C:\\Users\\bhabu\\Desktop\\Bitirme\\FF2\\original_sequences\\youtube\\c23\\videos\\org ('

dataset_fake_dir = 'C:\\Users\\bhabu\\Desktop\\Bitirme\\Proje\\Dataset\\Fake\\'
fake_videos_dir = 'C:\\Users\\bhabu\\Desktop\\Bitirme\\FF2\\manipulated_sequences\\Deepfakes\\c23\\videos\\fake ('

for j in range(1,1001):
    # Opens the Video file
    cap= cv2.VideoCapture(real_videos_dir+str(j)+').mp4')
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    i=0
    while(i <= frame_count):
        i+= round(frame_count / 3)
        for a in range(round(frame_count / 3)):
            ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite(dataset_real_dir+str(j)+'_'+str(i)+'.jpg',frame)
    
    cap.release()
    cv2.destroyAllWindows()
    
for j in range(1,1001):
    # Opens the Video file
    cap= cv2.VideoCapture(fake_videos_dir+str(j)+').mp4')
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    i=0
    while(i <= frame_count):
        i+= round(frame_count / 3)
        for a in range(round(frame_count / 3)):
            ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite(dataset_fake_dir+str(j)+'_'+str(i)+'.jpg',frame)
    
    cap.release()
    cv2.destroyAllWindows()