using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using UnityEngine.UI;

public class ImageLoader : MonoBehaviour
{
    public RawImage rawimage;
    public float ImageWidth;
    public float ImageHeight;

    //private static ImageLoader instance;
    
    // Start is called before the first frame update
    void Awake() 
    {
        //instance = this;
    }

    void Start()
    {
        rawimage.color = Color.black;
    }

    public void changeImageSize(float width, float height)
    {
        ImageWidth = width;
        ImageHeight = height;
        rawimage.rectTransform.sizeDelta = new Vector2(ImageWidth, ImageHeight);
    }

    public void changeColor(Color value)
    {
        rawimage.color = value;
    }

    public void setTexture(WebCamTexture value)
    {
        rawimage.texture = value;
    }

    public void setMaterial(Material value)
    {
        rawimage.material = value;
    }

    public void setMaterialTexture(Texture2D value)
    {
        rawimage.material.mainTexture = value;
    }

    public void removeTexture()
    {
        rawimage.texture = null;
    }

    public void removeAllTexture()
    {
        rawimage.texture = null;
        rawimage.material.mainTexture = null;
    }

    public void showTexture()
    {
        changeColor(Color.white);
    }

    public void hideTexture()
    {
        changeColor(Color.black);
    }

    public void RotateImage()
    {
        Debug.Log(rawimage.transform.eulerAngles.z);
        var newZ = 0;
        if(rawimage.transform.eulerAngles.z == newZ)
        {
            newZ = -90;
        }
        rawimage.transform.eulerAngles = new Vector3(0, 0, newZ);
    }
}
