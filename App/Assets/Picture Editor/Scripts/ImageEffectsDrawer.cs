using System.IO;
using UnityEngine;


[System.Serializable]
public class ImageEffect
{
    public MeshRenderer sourceMesh;
    public Texture2D sourceTex;

    #region CondenceTexture
    /// <summary>
    /// Compresses Image
    /// </summary>
    /// <param name="sourceTex">Original texture</param>
    /// <param name="constrain">Max pixels picture can be either width or height</param>
    /// <returns></returns>
    public void CondenceTexture(int constrain)
    {
        int width = (sourceTex.height > sourceTex.width) ? (sourceTex.width * constrain) / sourceTex.height : constrain;
        int height = (sourceTex.height > sourceTex.width) ? constrain : (sourceTex.height * constrain) / sourceTex.width;

        float pW = (float)width / (float)sourceTex.width;
        float pH = (float)height / (float)sourceTex.height;
        Texture2D nTex = new Texture2D((int)width, (int)height);
        for (int i = 0; i < sourceTex.height * sourceTex.width; i += (int)(1 / pW))
        {
            int y = Mathf.FloorToInt(((float)i) / ((float)sourceTex.width));
            int x = Mathf.FloorToInt(((float)i - ((float)(y * sourceTex.width))));

            int nX = Mathf.FloorToInt((((float)x) * ((float)pW)));
            int nY = Mathf.FloorToInt((((float)y) * ((float)pH)));

            Color c = sourceTex.GetPixel(x, y);

            nTex.SetPixel(nX, nY, c);
        }

        nTex.Apply();

        sourceMesh.sharedMaterial.mainTexture = nTex;
    }
    #endregion

    #region ToCircle
    /// <summary>
    /// Converts Image to a circle
    /// </summary>
    /// <param name="sourceTex">Original texture</param>
    /// <returns></returns>
    public void ToCircle()
    {
        int r = (sourceTex.height < sourceTex.width) ? sourceTex.height / 2 : sourceTex.width / 2;
        int h = (int)r * 2;
        int w = (int)r * 2;
        int Sx = (sourceTex.height > sourceTex.width) ? 0 : (int)((sourceTex.width - (r * 2)) / 2);
        int Sy = (sourceTex.height > sourceTex.width) ? (int)((sourceTex.height - (r * 2)) / 2) : 0;
        int cx = sourceTex.width / 2;
        int cy = sourceTex.height / 2;
        int Ex = Sx + (r * 2);
        int Ey = Sy + (r * 2);


        Texture2D b = new Texture2D(w, h);
        for (int i = 0; i < sourceTex.height * sourceTex.width; i++)
        {
            int y = Mathf.FloorToInt(((float)i) / ((float)sourceTex.width));
            int x = Mathf.FloorToInt(((float)i - ((float)(y * sourceTex.width))));
            Color c = sourceTex.GetPixel(x, y);
            if (y >= Sy && y <= Ey && x >= Sx && x <= Ex)
            {
                if (r * r >= (x - cx) * (x - cx) + (y - cy) * (y - cy))
                {
                    b.SetPixel(x - Sx, y - Sy, c);
                }
                else
                {
                    b.SetPixel(x - Sx, y - Sy, Color.clear);
                }
            }
        }
        b.Apply();
        sourceMesh.sharedMaterial.mainTexture = b;
    }
    #endregion

    #region ToSquare
    /// <summary>
    /// Converts Image to a square
    /// </summary>
    /// <param name="sourceTex">Original texture</param>
    /// <returns></returns>
    public void ToSquare()
    {
        int r = (sourceTex.height < sourceTex.width) ? sourceTex.height / 2 : sourceTex.width / 2;
        int h = (int)r * 2;
        int w = (int)r * 2;
        int Sx = (sourceTex.height > sourceTex.width) ? 0 : (int)((sourceTex.width - (r * 2)) / 2);
        int Sy = (sourceTex.height > sourceTex.width) ? (int)((sourceTex.height - (r * 2)) / 2) : 0;
        int cx = sourceTex.width / 2;
        int cy = sourceTex.height / 2;
        int Ex = Sx + (r * 2);
        int Ey = Sy + (r * 2);


        Texture2D b = new Texture2D(w, h);
        for (int i = 0; i < sourceTex.height * sourceTex.width; i++)
        {
            int y = Mathf.FloorToInt(((float)i) / ((float)sourceTex.width));
            int x = Mathf.FloorToInt(((float)i - ((float)(y * sourceTex.width))));
            Color c = sourceTex.GetPixel(x, y);
            if (y >= Sy && y <= Ey && x >= Sx && x <= Ex)
            {
                b.SetPixel(x - Sx, y - Sy, c);
            }
        }
        b.Apply();
        sourceMesh.sharedMaterial.mainTexture = b;
    }
    #endregion

    #region ToEllipse
    /// <summary>
    /// ConvertsImage to an ellipse
    /// </summary>
    /// <param name="sourceTex">Original texture</param>
    /// <returns></returns>
    public void ToEllipse()
    {
        int rW = sourceTex.width / 2;
        int rH = sourceTex.height / 2;
        int cx = sourceTex.width / 2;
        int cy = sourceTex.height / 2;


        Texture2D b = new Texture2D(sourceTex.width, sourceTex.height);
        for (int i = 0; i < sourceTex.height * sourceTex.width; i++)
        {
            int y = Mathf.FloorToInt(((float)i) / ((float)sourceTex.width));
            int x = Mathf.FloorToInt(((float)i - ((float)(y * sourceTex.width))));
            Color c = sourceTex.GetPixel(x, y);
            if (1 >= (float)((x - cx) * (x - cx)) / (float)(rW * rW) + (float)((y - cy) * (y - cy)) / (float)(rH * rH))
            {
                b.SetPixel(x, y, c);
            }
            else
            {
                b.SetPixel(x, y, Color.clear);
            }
        }
        b.Apply();
        sourceMesh.sharedMaterial.mainTexture = b;
    }
    #endregion

    #region CustomCutout
    /// <summary>
    /// Cuts out a custom image
    /// </summary>
    /// <param name="sourceTex">Original texture</param>
    /// <param name="cutout">Cutout reference image</param>
    /// <returns></returns>
    public void CustomCutout(Texture2D cutout)
    {
        bool autoFit = true; //Don't change to false, I am still developing autoFit
        float cutoutHToWAspect = (float)(cutout.height / cutout.width);
        float sourceHToWAspect = (float)(sourceTex.height / sourceTex.width);
        float sourceToCutoutAspect = (cutoutHToWAspect > sourceHToWAspect) ? ((float)sourceTex.height / (float)cutout.height) : (float)(sourceTex.width / (float)cutout.width);
        int h = (autoFit && cutoutHToWAspect > sourceHToWAspect) ? sourceTex.height : (autoFit) ? (int)(sourceTex.width * cutoutHToWAspect) : cutout.height;
        int w = (autoFit && sourceHToWAspect > cutoutHToWAspect) ? sourceTex.width : (autoFit) ? (int)(sourceTex.height / cutoutHToWAspect) : cutout.width;
        int Sx = (sourceTex.width - w) / 2;
        int Sy = (sourceTex.height - h) / 2;
        int Ex = Sx + w;
        int Ey = Sy + h;

        Texture2D b = new Texture2D(h, w);
        for (int i = 0; i < sourceTex.height * sourceTex.width; i++)
        {
            int y = Mathf.FloorToInt(((float)i) / ((float)sourceTex.width));
            int x = Mathf.FloorToInt(((float)i - ((float)(y * sourceTex.width))));
            Color c = sourceTex.GetPixel(x, y);
            if (y >= Sy && y <= Ey && x >= Sx && x <= Ex)
            {
                if (cutout.GetPixel(x - Sx, y - Sy) == Color.white && !autoFit)
                {
                    b.SetPixel(x - Sx, y - Sy, c);
                }
                else if (cutout.GetPixel(Mathf.RoundToInt((float)(x - Sx) / (float)sourceToCutoutAspect), Mathf.RoundToInt((float)(y - Sy) / (float)sourceToCutoutAspect)) == Color.white && autoFit)
                {
                    b.SetPixel(x - Sx, y - Sy, c);
                }
                else
                {
                    b.SetPixel(x - Sx, y - Sy, Color.clear);
                }
            }
        }
        b.Apply();
        sourceMesh.sharedMaterial.mainTexture = b;
    }
    #endregion





    #region ToCircle
    /// <summary>
    /// Converts Image to a circle
    /// </summary>
    /// <param name="sourceTex">Original texture</param>
    /// <returns></returns>
    public void ToCircle(int constrain)
    {
        Texture2D sourceTex = GrabCondenceTexture(constrain);

        int r = (sourceTex.height < sourceTex.width) ? sourceTex.height / 2 : sourceTex.width / 2;
        int h = (int)r * 2;
        int w = (int)r * 2;
        int Sx = (sourceTex.height > sourceTex.width) ? 0 : (int)((sourceTex.width - (r * 2)) / 2);
        int Sy = (sourceTex.height > sourceTex.width) ? (int)((sourceTex.height - (r * 2)) / 2) : 0;
        int cx = sourceTex.width / 2;
        int cy = sourceTex.height / 2;
        int Ex = Sx + (r * 2);
        int Ey = Sy + (r * 2);


        Texture2D b = new Texture2D(w, h);
        for (int i = 0; i < sourceTex.height * sourceTex.width; i++)
        {
            int y = Mathf.FloorToInt(((float)i) / ((float)sourceTex.width));
            int x = Mathf.FloorToInt(((float)i - ((float)(y * sourceTex.width))));
            Color c = sourceTex.GetPixel(x, y);
            if (y >= Sy && y <= Ey && x >= Sx && x <= Ex)
            {
                if (r * r >= (x - cx) * (x - cx) + (y - cy) * (y - cy))
                {
                    b.SetPixel(x - Sx, y - Sy, c);
                }
                else
                {
                    b.SetPixel(x - Sx, y - Sy, Color.clear);
                }
            }
        }
        b.Apply();
        sourceMesh.sharedMaterial.mainTexture = b;
    }
    #endregion

    #region ToSquare
    /// <summary>
    /// Converts Image to a square
    /// </summary>
    /// <param name="sourceTex">Original texture</param>
    /// <returns></returns>
    public void ToSquare(int constrain)
    {
        Texture2D sourceTex = GrabCondenceTexture(constrain);

        int r = (sourceTex.height < sourceTex.width) ? sourceTex.height / 2 : sourceTex.width / 2;
        int h = (int)r * 2;
        int w = (int)r * 2;
        int Sx = (sourceTex.height > sourceTex.width) ? 0 : (int)((sourceTex.width - (r * 2)) / 2);
        int Sy = (sourceTex.height > sourceTex.width) ? (int)((sourceTex.height - (r * 2)) / 2) : 0;
        int cx = sourceTex.width / 2;
        int cy = sourceTex.height / 2;
        int Ex = Sx + (r * 2);
        int Ey = Sy + (r * 2);


        Texture2D b = new Texture2D(w, h);
        for (int i = 0; i < sourceTex.height * sourceTex.width; i++)
        {
            int y = Mathf.FloorToInt(((float)i) / ((float)sourceTex.width));
            int x = Mathf.FloorToInt(((float)i - ((float)(y * sourceTex.width))));
            Color c = sourceTex.GetPixel(x, y);
            if (y >= Sy && y <= Ey && x >= Sx && x <= Ex)
            {
                b.SetPixel(x - Sx, y - Sy, c);
            }
        }
        b.Apply();
        sourceMesh.sharedMaterial.mainTexture = b;
    }
    #endregion

    #region ToEllipse
    /// <summary>
    /// ConvertsImage to an ellipse
    /// </summary>
    /// <param name="sourceTex">Original texture</param>
    /// <returns></returns>
    public void ToEllipse(int constrain)
    {
        Texture2D sourceTex = GrabCondenceTexture(constrain);

        int rW = sourceTex.width / 2;
        int rH = sourceTex.height / 2;
        int cx = sourceTex.width / 2;
        int cy = sourceTex.height / 2;


        Texture2D b = new Texture2D(sourceTex.width, sourceTex.height);
        for (int i = 0; i < sourceTex.height * sourceTex.width; i++)
        {
            int y = Mathf.FloorToInt(((float)i) / ((float)sourceTex.width));
            int x = Mathf.FloorToInt(((float)i - ((float)(y * sourceTex.width))));
            Color c = sourceTex.GetPixel(x, y);
            if (1 >= (float)((x - cx) * (x - cx)) / (float)(rW * rW) + (float)((y - cy) * (y - cy)) / (float)(rH * rH))
            {
                b.SetPixel(x, y, c);
            }
            else
            {
                b.SetPixel(x, y, Color.clear);
            }
        }
        b.Apply();
        sourceMesh.sharedMaterial.mainTexture = b;
    }
    #endregion

    #region CustomCutout
    /// <summary>
    /// Cuts out a custom image
    /// </summary>
    /// <param name="sourceTex">Original texture</param>
    /// <param name="cutout">Cutout reference image</param>
    /// <returns></returns>
    public void CustomCutout(Texture2D cutout, int constrain)
    {
        Texture2D sourceTex = GrabCondenceTexture(constrain);

        bool autoFit = true; //Don't change to false, I am still developing autoFit
        float cutoutHToWAspect = (float)(cutout.height / cutout.width);
        float sourceHToWAspect = (float)(sourceTex.height / sourceTex.width);
        float sourceToCutoutAspect = (cutoutHToWAspect > sourceHToWAspect) ? ((float)sourceTex.height / (float)cutout.height) : (float)(sourceTex.width / (float)cutout.width);
        int h = (autoFit && cutoutHToWAspect > sourceHToWAspect) ? sourceTex.height : (autoFit) ? (int)(sourceTex.width * cutoutHToWAspect) : cutout.height;
        int w = (autoFit && sourceHToWAspect > cutoutHToWAspect) ? sourceTex.width : (autoFit) ? (int)(sourceTex.height / cutoutHToWAspect) : cutout.width;
        int Sx = (sourceTex.width - w) / 2;
        int Sy = (sourceTex.height - h) / 2;
        int Ex = Sx + w;
        int Ey = Sy + h;

        Texture2D b = new Texture2D(h, w);
        for (int i = 0; i < sourceTex.height * sourceTex.width; i++)
        {
            int y = Mathf.FloorToInt(((float)i) / ((float)sourceTex.width));
            int x = Mathf.FloorToInt(((float)i - ((float)(y * sourceTex.width))));
            Color c = sourceTex.GetPixel(x, y);
            if (y >= Sy && y <= Ey && x >= Sx && x <= Ex)
            {
                if (cutout.GetPixel(x - Sx, y - Sy) == Color.white && !autoFit)
                {
                    b.SetPixel(x - Sx, y - Sy, c);
                }
                else if (cutout.GetPixel(Mathf.RoundToInt((float)(x - Sx) / (float)sourceToCutoutAspect), Mathf.RoundToInt((float)(y - Sy) / (float)sourceToCutoutAspect)) == Color.white && autoFit)
                {
                    b.SetPixel(x - Sx, y - Sy, c);
                }
                else
                {
                    b.SetPixel(x - Sx, y - Sy, Color.clear);
                }
            }
        }
        b.Apply();
        sourceMesh.sharedMaterial.mainTexture = b;
    }
    #endregion






    #region GrabCondenceTexture
    /// <summary>
    /// Compresses Image And Returns New Image
    /// </summary>
    /// <param name="sourceTex">Original texture</param>
    /// <param name="constrain">Max pixels picture can be either width or height</param>
    /// <returns></returns>
    public Texture2D GrabCondenceTexture(int constrain)
    {
        int width = (sourceTex.height > sourceTex.width) ? (sourceTex.width * constrain) / sourceTex.height : constrain;
        int height = (sourceTex.height > sourceTex.width) ? constrain : (sourceTex.height * constrain) / sourceTex.width;

        float pW = (float)width / (float)sourceTex.width;
        float pH = (float)height / (float)sourceTex.height;
        Texture2D nTex = new Texture2D((int)width, (int)height);
        for (int i = 0; i < sourceTex.height * sourceTex.width; i += (int)(1 / pW))
        {
            int y = Mathf.FloorToInt(((float)i) / ((float)sourceTex.width));
            int x = Mathf.FloorToInt(((float)i - ((float)(y * sourceTex.width))));

            int nX = Mathf.FloorToInt((((float)x) * ((float)pW)));
            int nY = Mathf.FloorToInt((((float)y) * ((float)pH)));

            Color c = sourceTex.GetPixel(x, y);

            nTex.SetPixel(nX, nY, c);
        }

        nTex.Apply();

        return nTex;
    }
    #endregion

    #region GrabToCircle
    /// <summary>
    /// Converts Image to a circle And Returns New Image
    /// </summary>
    /// <param name="sourceTex">Original texture</param>
    /// <returns></returns>
    public Texture2D GrabToCircle()
    {
        int r = (sourceTex.height < sourceTex.width) ? sourceTex.height / 2 : sourceTex.width / 2;
        int h = (int)r * 2;
        int w = (int)r * 2;
        int Sx = (sourceTex.height > sourceTex.width) ? 0 : (int)((sourceTex.width - (r * 2)) / 2);
        int Sy = (sourceTex.height > sourceTex.width) ? (int)((sourceTex.height - (r * 2)) / 2) : 0;
        int cx = sourceTex.width / 2;
        int cy = sourceTex.height / 2;
        int Ex = Sx + (r * 2);
        int Ey = Sy + (r * 2);


        Texture2D b = new Texture2D(w, h);
        for (int i = 0; i < sourceTex.height * sourceTex.width; i++)
        {
            int y = Mathf.FloorToInt(((float)i) / ((float)sourceTex.width));
            int x = Mathf.FloorToInt(((float)i - ((float)(y * sourceTex.width))));
            Color c = sourceTex.GetPixel(x, y);
            if (y >= Sy && y <= Ey && x >= Sx && x <= Ex)
            {
                if (r * r >= (x - cx) * (x - cx) + (y - cy) * (y - cy))
                {
                    b.SetPixel(x - Sx, y - Sy, c);
                }
                else
                {
                    b.SetPixel(x - Sx, y - Sy, Color.clear);
                }
            }
        }
        b.Apply();
        return b;
    }
    #endregion

    #region GrabToSquare
    /// <summary>
    /// Converts Image to a square And Returns New Image
    /// </summary>
    /// <param name="sourceTex">Original texture</param>
    /// <returns></returns>
    public Texture2D GrabToSquare()
    {
        int r = (sourceTex.height < sourceTex.width) ? sourceTex.height / 2 : sourceTex.width / 2;
        int h = (int)r * 2;
        int w = (int)r * 2;
        int Sx = (sourceTex.height > sourceTex.width) ? 0 : (int)((sourceTex.width - (r * 2)) / 2);
        int Sy = (sourceTex.height > sourceTex.width) ? (int)((sourceTex.height - (r * 2)) / 2) : 0;
        int cx = sourceTex.width / 2;
        int cy = sourceTex.height / 2;
        int Ex = Sx + (r * 2);
        int Ey = Sy + (r * 2);


        Texture2D b = new Texture2D(w, h);
        for (int i = 0; i < sourceTex.height * sourceTex.width; i++)
        {
            int y = Mathf.FloorToInt(((float)i) / ((float)sourceTex.width));
            int x = Mathf.FloorToInt(((float)i - ((float)(y * sourceTex.width))));
            Color c = sourceTex.GetPixel(x, y);
            if (y >= Sy && y <= Ey && x >= Sx && x <= Ex)
            {
                b.SetPixel(x - Sx, y - Sy, c);
            }
        }
        b.Apply();
        return b;
    }
    #endregion

    #region GrabToEllipse
    /// <summary>
    /// ConvertsImage to an ellipse And Returns New Image
    /// </summary>
    /// <param name="sourceTex">Original texture</param>
    /// <returns></returns>
    public Texture2D GrabToEllipse()
    {
        int rW = sourceTex.width / 2;
        int rH = sourceTex.height / 2;
        int cx = sourceTex.width / 2;
        int cy = sourceTex.height / 2;


        Texture2D b = new Texture2D(sourceTex.width, sourceTex.height);
        for (int i = 0; i < sourceTex.height * sourceTex.width; i++)
        {
            int y = Mathf.FloorToInt(((float)i) / ((float)sourceTex.width));
            int x = Mathf.FloorToInt(((float)i - ((float)(y * sourceTex.width))));
            Color c = sourceTex.GetPixel(x, y);
            if (1 >= (float)((x - cx) * (x - cx)) / (float)(rW * rW) + (float)((y - cy) * (y - cy)) / (float)(rH * rH))
            {
                b.SetPixel(x, y, c);
            }
            else
            {
                b.SetPixel(x, y, Color.clear);
            }
        }
        b.Apply();
        return b;
    }
    #endregion

    #region GrabCustomCutout
    /// <summary>
    /// Cuts out a custom image And Returns New Image
    /// </summary>
    /// <param name="sourceTex">Original texture</param>
    /// <param name="cutout">Cutout reference image</param>
    /// <returns></returns>
    public Texture2D GrabCustomCutout(Texture2D cutout)
    {
        bool autoFit = true;
        float cutoutHToWAspect = (float)(cutout.height / cutout.width);
        float sourceHToWAspect = (float)(sourceTex.height / sourceTex.width);
        float sourceToCutoutAspect = (cutoutHToWAspect > sourceHToWAspect) ? ((float)sourceTex.height / (float)cutout.height) : (float)(sourceTex.width / (float)cutout.width);
        int h = (autoFit && cutoutHToWAspect > sourceHToWAspect) ? sourceTex.height : (autoFit) ? (int)(sourceTex.width * cutoutHToWAspect) : cutout.height;
        int w = (autoFit && sourceHToWAspect > cutoutHToWAspect) ? sourceTex.width : (autoFit) ? (int)(sourceTex.height / cutoutHToWAspect) : cutout.width;
        int Sx = (sourceTex.width - w) / 2;
        int Sy = (sourceTex.height - h) / 2;
        int Ex = Sx + w;
        int Ey = Sy + h;

        Texture2D b = new Texture2D(h, w);
        for (int i = 0; i < sourceTex.height * sourceTex.width; i++)
        {
            int y = Mathf.FloorToInt(((float)i) / ((float)sourceTex.width));
            int x = Mathf.FloorToInt(((float)i - ((float)(y * sourceTex.width))));
            Color c = sourceTex.GetPixel(x, y);
            if (y >= Sy && y <= Ey && x >= Sx && x <= Ex)
            {
                if (cutout.GetPixel(x - Sx, y - Sy) == Color.white && !autoFit)
                {
                    b.SetPixel(x - Sx, y - Sy, c);
                }
                else if (cutout.GetPixel(Mathf.RoundToInt((float)(x - Sx) / (float)sourceToCutoutAspect), Mathf.RoundToInt((float)(y - Sy) / (float)sourceToCutoutAspect)) == Color.white && autoFit)
                {
                    b.SetPixel(x - Sx, y - Sy, c);
                }
                else
                {
                    b.SetPixel(x - Sx, y - Sy, Color.clear);
                }
            }
        }
        b.Apply();
        return b;
    }
    #endregion

    #region GrabToCircle
    /// <summary>
    /// Converts Image to a circle And Returns New Image
    /// </summary>
    /// <param name="sourceTex">Original texture</param>
    /// <returns></returns>
    public Texture2D GrabToCircle(int constrain)
    {
        Texture2D sourceTex = GrabCondenceTexture(constrain);

        int r = (sourceTex.height < sourceTex.width) ? sourceTex.height / 2 : sourceTex.width / 2;
        int h = (int)r * 2;
        int w = (int)r * 2;
        int Sx = (sourceTex.height > sourceTex.width) ? 0 : (int)((sourceTex.width - (r * 2)) / 2);
        int Sy = (sourceTex.height > sourceTex.width) ? (int)((sourceTex.height - (r * 2)) / 2) : 0;
        int cx = sourceTex.width / 2;
        int cy = sourceTex.height / 2;
        int Ex = Sx + (r * 2);
        int Ey = Sy + (r * 2);


        Texture2D b = new Texture2D(w, h);
        for (int i = 0; i < sourceTex.height * sourceTex.width; i++)
        {
            int y = Mathf.FloorToInt(((float)i) / ((float)sourceTex.width));
            int x = Mathf.FloorToInt(((float)i - ((float)(y * sourceTex.width))));
            Color c = sourceTex.GetPixel(x, y);
            if (y >= Sy && y <= Ey && x >= Sx && x <= Ex)
            {
                if (r * r >= (x - cx) * (x - cx) + (y - cy) * (y - cy))
                {
                    b.SetPixel(x - Sx, y - Sy, c);
                }
                else
                {
                    b.SetPixel(x - Sx, y - Sy, Color.clear);
                }
            }
        }
        b.Apply();
        return b;
    }
    #endregion

    #region GrabToSquare
    /// <summary>
    /// Converts Image to a square And Returns New Image
    /// </summary>
    /// <param name="sourceTex">Original texture</param>
    /// <returns></returns>
    public Texture2D GrabToSquare(int constrain)
    {
        Texture2D sourceTex = GrabCondenceTexture(constrain);

        int r = (sourceTex.height < sourceTex.width) ? sourceTex.height / 2 : sourceTex.width / 2;
        int h = (int)r * 2;
        int w = (int)r * 2;
        int Sx = (sourceTex.height > sourceTex.width) ? 0 : (int)((sourceTex.width - (r * 2)) / 2);
        int Sy = (sourceTex.height > sourceTex.width) ? (int)((sourceTex.height - (r * 2)) / 2) : 0;
        int cx = sourceTex.width / 2;
        int cy = sourceTex.height / 2;
        int Ex = Sx + (r * 2);
        int Ey = Sy + (r * 2);


        Texture2D b = new Texture2D(w, h);
        for (int i = 0; i < sourceTex.height * sourceTex.width; i++)
        {
            int y = Mathf.FloorToInt(((float)i) / ((float)sourceTex.width));
            int x = Mathf.FloorToInt(((float)i - ((float)(y * sourceTex.width))));
            Color c = sourceTex.GetPixel(x, y);
            if (y >= Sy && y <= Ey && x >= Sx && x <= Ex)
            {
                b.SetPixel(x - Sx, y - Sy, c);
            }
        }
        b.Apply();
        return b;
    }
    #endregion

    #region GrabToEllipse
    /// <summary>
    /// ConvertsImage to an ellipse And Returns New Image
    /// </summary>
    /// <param name="sourceTex">Original texture</param>
    /// <returns></returns>
    public Texture2D GrabToEllipse(int constrain)
    {
        Texture2D sourceTex = GrabCondenceTexture(constrain);

        int rW = sourceTex.width / 2;
        int rH = sourceTex.height / 2;
        int cx = sourceTex.width / 2;
        int cy = sourceTex.height / 2;


        Texture2D b = new Texture2D(sourceTex.width, sourceTex.height);
        for (int i = 0; i < sourceTex.height * sourceTex.width; i++)
        {
            int y = Mathf.FloorToInt(((float)i) / ((float)sourceTex.width));
            int x = Mathf.FloorToInt(((float)i - ((float)(y * sourceTex.width))));
            Color c = sourceTex.GetPixel(x, y);
            if (1 >= (float)((x - cx) * (x - cx)) / (float)(rW * rW) + (float)((y - cy) * (y - cy)) / (float)(rH * rH))
            {
                b.SetPixel(x, y, c);
            }
            else
            {
                b.SetPixel(x, y, Color.clear);
            }
        }
        b.Apply();
        return b;
    }
    #endregion

    #region GrabCustomCutout
    /// <summary>
    /// Cuts out a custom image And Returns New Image
    /// </summary>
    /// <param name="sourceTex">Original texture</param>
    /// <param name="cutout">Cutout reference image</param>
    /// <returns></returns>
    public Texture2D GrabCustomCutout(Texture2D cutout, int constrain)
    {
        Texture2D sourceTex = GrabCondenceTexture(constrain);

        bool autoFit = true;
        float cutoutHToWAspect = (float)(cutout.height / cutout.width);
        float sourceHToWAspect = (float)(sourceTex.height / sourceTex.width);
        float sourceToCutoutAspect = (cutoutHToWAspect > sourceHToWAspect) ? ((float)sourceTex.height / (float)cutout.height) : (float)(sourceTex.width / (float)cutout.width);
        int h = (autoFit && cutoutHToWAspect > sourceHToWAspect) ? sourceTex.height : (autoFit) ? (int)(sourceTex.width * cutoutHToWAspect) : cutout.height;
        int w = (autoFit && sourceHToWAspect > cutoutHToWAspect) ? sourceTex.width : (autoFit) ? (int)(sourceTex.height / cutoutHToWAspect) : cutout.width;
        int Sx = (sourceTex.width - w) / 2;
        int Sy = (sourceTex.height - h) / 2;
        int Ex = Sx + w;
        int Ey = Sy + h;

        Texture2D b = new Texture2D(h, w);
        for (int i = 0; i < sourceTex.height * sourceTex.width; i++)
        {
            int y = Mathf.FloorToInt(((float)i) / ((float)sourceTex.width));
            int x = Mathf.FloorToInt(((float)i - ((float)(y * sourceTex.width))));
            Color c = sourceTex.GetPixel(x, y);
            if (y >= Sy && y <= Ey && x >= Sx && x <= Ex)
            {
                if (cutout.GetPixel(x - Sx, y - Sy) == Color.white && !autoFit)
                {
                    b.SetPixel(x - Sx, y - Sy, c);
                }
                else if (cutout.GetPixel(Mathf.RoundToInt((float)(x - Sx) / (float)sourceToCutoutAspect), Mathf.RoundToInt((float)(y - Sy) / (float)sourceToCutoutAspect)) == Color.white && autoFit)
                {
                    b.SetPixel(x - Sx, y - Sy, c);
                }
                else
                {
                    b.SetPixel(x - Sx, y - Sy, Color.clear);
                }
            }
        }
        b.Apply();
        return b;
    }
    #endregion

    public void set(Texture2D tex)
    {
        sourceMesh.sharedMaterial.mainTexture = tex;
    }

    public void saveFile(string fileName, Texture2D tex)
    {
        // Encode texture into PNG
        byte[] bytes = tex.EncodeToPNG();
        
        File.WriteAllBytes(Application.dataPath + "/" + fileName +".png", bytes);
    }


}
