import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;
import java.awt.Graphics;
import java.awt.image.ColorModel;
import java.awt.image.IndexColorModel;
import java.awt.image.WritableRaster;


import javax.imageio.ImageIO;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.Raster;
import java.io.File;
import java.io.IOException;
@SuppressWarnings("unused")
public class Main  {
	
	static BufferedImage image2;
	 static File f = null;
	public static void main(String[] args) {
		try
		{
			//Se carga la imagen
			String fn = args[0];
			System.out.println(fn);
			File sf = new File(fn);
			BufferedImage image1 = ImageIO.read(sf);
			int width = image1.getWidth();
			int height = image1.getHeight();
			
			//Se convierte a Blanco y Negro
			for(int i=0; i<height; i++){
		         
	            for(int j=0; j<width; j++){
	            
	               Color c = new Color(image1.getRGB(j, i));
	               int red = (int)(c.getRed() * 0.299);
	               int green = (int)(c.getGreen() * 0.587);
	               int blue = (int)(c.getBlue() *0.114);
	               int rgb = range(red+green+blue,8);
	               Color newColor = new Color(rgb,rgb,rgb);

	               
	               image1.setRGB(j,i,newColor.getRGB());
	            }
	         }
			
			try{
			      f = new File("saved" + sf.getName());
			      image1 = convertRGBAToIndexed(image1);
			      ImageIO.write(image1, "bmp", f);
			    }catch(IOException e){
			      System.out.println(e);
			    }

		}
		catch(IOException e)
		{
			System.out.print("No");
		}		
		
	}
	
	public static int range(int n, double prob) {
		double res = ((100 * prob)/10);
		
		int[]array = new int[(int)res];
		array[0]= 1;
		array[1]=255;
		
		for (int i = 2 ; i <= res - 2; i++)
		{
			array[i] = n;
		}
	    int rnd = new Random().nextInt(array.length);
	    return array[rnd];
	}

	public static int fromColor(Color c) {
    	return ((c.getAlpha() >> 6) << 6)
         + ((c.getRed()   >> 6) << 4)
         + ((c.getGreen() >> 6) << 2)
         +  (c.getBlue()  >> 6);
  	}

	public static Color toColor(int i) {
	    return new Color(((i >> 4) % 4) * 64,
	                     ((i >> 2) % 4) * 64,
	                      (i       % 4) * 64,
	                      (i >> 6)      * 64);
	}

	public static BufferedImage convertRGBAToIndexed(BufferedImage src) {
	    BufferedImage dest = new BufferedImage(src.getWidth(), src.getHeight(), BufferedImage.TYPE_BYTE_INDEXED);
	    Graphics g = dest.getGraphics();
	    g.setColor(new Color(231, 20, 189));

	    // fill with a hideous color and make it transparent
	    g.fillRect(0, 0, dest.getWidth(), dest.getHeight());
	    dest = makeTransparent(dest, 0, 0);

	    dest.createGraphics().drawImage(src, 0, 0, null);
	    return dest;
	}

	public static BufferedImage makeTransparent(BufferedImage image, int x, int y) {
	    ColorModel cm = image.getColorModel();
	    if (!(cm instanceof IndexColorModel))
	        return image; // sorry...
	    IndexColorModel icm = (IndexColorModel) cm;
	    WritableRaster raster = image.getRaster();
	    int pixel = raster.getSample(x, y, 0); // pixel is offset in ICM's palette
	    int size = icm.getMapSize();
	    byte[] reds = new byte[size];
	    byte[] greens = new byte[size];
	    byte[] blues = new byte[size];
	    icm.getReds(reds);
	    icm.getGreens(greens);
	    icm.getBlues(blues);
	    IndexColorModel icm2 = new IndexColorModel(8, size, reds, greens, blues, pixel);
	    return new BufferedImage(icm2, raster, image.isAlphaPremultiplied(), null);
	}
	
}

	

