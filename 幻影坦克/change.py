from PIL import Image
import time
def colorful_shio(brightness_f,brightness_b):
    '''给定两个参数调整里外图亮度以求得最佳幻坦效果'''
    start=time.time()
    image_f=Image.open('1.jpg')
    image_b=Image.open('2.jpg')
    w_f,h_f=image_f.size
    w_b,h_b=image_b.size
    w_min=min(w_f,w_b)
    h_min=min(h_f,h_b)
    new_image=Image.new('RGBA',(w_min,h_min))
    array_f=image_f.load()
    array_b=image_b.load()
    scale_h_f=int(h_f/h_min)
    scale_w_f=int(w_f/w_min)
    scale_h_b=int(h_b/h_min)
    scale_w_b=int(w_b/w_min)
    scale_f=min(scale_h_f,scale_w_f)
    scale_b=min(scale_h_b,scale_w_b)
    trans_f_x=int((w_f-w_min*scale_f)/2)
    trans_b_x=int((w_b-w_min*scale_b)/2)
    #10-8，11-7，11-8
    a=brightness_f
    b=brightness_b
    for i in range(0,w_min):
        for j in range(0,h_min):
            R_f, G_f, B_f=array_f[trans_f_x+i*scale_f,j*scale_f]
            R_b, G_b,B_b=array_b[trans_b_x+i*scale_b,j*scale_b]
            R_f *= a/10
            R_b *= b/10
            G_f *= a/10
            G_b *= b/10
            B_f *= a/10
            B_b *= b/10
            delta_r = R_b - R_f
            delta_g = G_b - G_f
            delta_b = B_b - B_f
            coe_a = 8+255/256+(delta_r - delta_b)/256
            coe_b = 4*delta_r + 8*delta_g + 6*delta_b + ((delta_r - delta_b)*(R_b+R_f))/256 + (delta_r**2 - delta_b**2)/512
            A_new = 255 + coe_b/(2*coe_a)
            A_new = int(A_new)
            if A_new<=0:
                A_new=0
                R_new=0
                G_new=0
                B_new=0
            elif A_new>=255:
                A_new=255
                R_new=int((255*(R_b)*b/10)/A_new)
                G_new=int((255*(G_b)*b/10)/A_new)
                B_new=int((255*(B_b)*b/10)/A_new)
            else:
                A_new=A_new
                R_new=int((255*(R_b)*b/10)/A_new)
                G_new=int((255*(G_b)*b/10)/A_new)
                B_new=int((255*(B_b)*b/10)/A_new)
            pixel_new=(R_new,G_new,B_new,A_new)
            new_image.putpixel((i,j),pixel_new)
    new_image.save('幻影坦克%d-%d.png'%(a,b))
    end=time.time()
    print('running time:%ds'%(end-start))
if __name__ == '__main__':
    colorful_shio(13,7)