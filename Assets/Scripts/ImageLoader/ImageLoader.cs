using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using UnityEngine.UI;

public class ImageLoader : MonoBehaviour
{
    public RawImage rawimage;
    public Image squareHover;
    public float ImageWidth;
    public float ImageHeight;
    public float HoverWidth;
    public float HoverHeight;

    private HoverController HoverScript;
    private float minX;
    private float maxX;
    private float minY;
    private float maxY;

    //private static ImageLoader instance;
    
    // Start is called before the first frame update
    void Awake() 
    {
        HoverScript = squareHover.GetComponent<HoverController>();
    }

    void Start()
    {
        rawimage.color = Color.black;
        //rawimage.SetNativeSize();
    }

    public void calculateHoverParam()
    {
        maxX = ImageWidth/2 - HoverWidth/2;
        minX = -maxX;
        maxY = ImageHeight/2 - HoverHeight/2;
        minY = -maxY;

        HoverScript.setHoverParam(ImageWidth, 0, ImageHeight, 0);  
    }

    public void ToggleHover(bool isOn)
    {
        squareHover.gameObject.SetActive(isOn);
    }

    public void ToggleHoverMovemnt(bool isOn)
    {
        HoverScript.setAllowMovement(isOn);
    }

    public Vector2 getHoverPosition()
    {
        Debug.Log(squareHover.transform.localPosition);
        return squareHover.transform.localPosition;
    }

    public Vector2 getHoverPositionInPixel()
    {
        Debug.Log(squareHover.transform.GetComponent<RectTransform>().anchoredPosition);
        return squareHover.transform.GetComponent<RectTransform>().anchoredPosition;
    }

    public void changeImageSize(float width, float height)
    {
        ImageWidth = height;
        ImageHeight = width;
        // rawimage.texture.width = (int)width;
        // rawimage.texture.height = (int)height;
        rawimage.rectTransform.sizeDelta = new Vector2(ImageWidth, ImageHeight);
        calculateHoverParam();
    }

    public void changeColor(Color value)
    {
        rawimage.color = value;
    }

    public void setTexture(WebCamTexture value)
    {
        rawimage.texture = value;
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
