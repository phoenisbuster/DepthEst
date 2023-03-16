using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using UnityEngine.UI;

public class BuiltInCameraFunctions : MonoBehaviour
{
    public RawImage rawimage;
    public ImageLoader ImageScript;
    public int DefaultCameraIndex = 0;
    public bool initWhenStart = false;
    private WebCamTexture webcamTexture;

    private static BuiltInCameraFunctions instance;

    private void Awake() 
    {
        instance = this;
    }

    public static BuiltInCameraFunctions getInstance()
    {
        return instance;
    }

    // Start is called before the first frame update
    void Start()
    {
        rawimage = ImageScript.rawimage;        
        if(initWhenStart)
        {
            initCamera();
        }
    }

    public void clickAccessCamera()
    {
        initCamera();
    }

    public void clickCapture()
    {
        StartCoroutine(SaveImage());
    } 

    public void changeCameraIndex()
    {
        DefaultCameraIndex = DefaultCameraIndex == 0? 1 : 0;
        webcamTexture.Stop();
        initCamera();
    }

    public void changeImageRotate()
    {
        ImageScript.RotateImage();
    }

    public void StopWebCamTex()
    {
        if(webcamTexture.isPlaying)
        {
            webcamTexture.Stop();
            rawimage.color = Color.black;
        }
    }

    public void PlayWebCamTex()
    {
        if(!webcamTexture.isPlaying)
        {
            webcamTexture.Play();
            rawimage.color = Color.white;
        }
    }

    private void initCamera()
    {
        DebugLog.getInstance().updateLog(LogType.WebCamText, "Init Camera Start", false);
 
        //Obtain camera devices available
        WebCamDevice[] cam_devices = WebCamTexture.devices;
        //Set a camera to the webcamTexture
        webcamTexture = new WebCamTexture(cam_devices[DefaultCameraIndex].name, 480, 480, 30);
        //Set the webcamTexture to the texture of the rawimage
        rawimage.color = Color.white;
        rawimage.texture = webcamTexture;
        //rawimage.material.mainTexture = webcamTexture;
        
        //Start the camera
        webcamTexture.Play();

        DebugLog.getInstance().updateLog(LogType.WebCamText, "Init Camera End");
    }

    private IEnumerator SaveImage()
    {
        DebugLog.getInstance().updateLog(LogType.WebCamText, "Save Pic Start", false);
        
        //Create a Texture2D with the size of the rendered image on the screen.
        Texture2D texture = new Texture2D(rawimage.texture.width, rawimage.texture.height, TextureFormat.ARGB32, false);
        //Save the image to the Texture2D
        texture.SetPixels(webcamTexture.GetPixels());
        //Rotate textture 90 degree to bypass the default option in Unity
        texture = RotateTexture(texture, -90);
        texture.Apply();

        yield return new WaitForEndOfFrame();

        // Save the screenshot to Gallery/Photos
        NativeGallery.Permission permission = NativeGallery.SaveImageToGallery(
                                                                    texture, 
                                                                    "CameraTest", 
                                                                    "CaptureImage.png", 
                                                                    (success, path) => 
                                                                    {
                                                                        Debug.Log(path);
                                                                        Debug.Log(path == "");
                                                                        DebugLog.getInstance().updateLog(
                                                                                                LogType.NativeGallery,
                                                                                                "Save Image To Gallery: " + 
                                                                                                success + " at " +
                                                                                                path 
                                                                        );
                                                                    }
        );
        SaveToResources.Save(texture.EncodeToPNG(), "TestWebCamText", "png");
        // To avoid memory leaks
        Destroy(texture);
    }

    private Texture2D RotateTexture (Texture2D texture, float eulerAngles)
    {
        int x;
        int y;
        int i;
        int j;
        float phi = eulerAngles / (180 / Mathf.PI);
        float sn = Mathf.Sin (phi);
        float cs = Mathf.Cos (phi);
        Color32[] arr = texture.GetPixels32 ();
        Color32[] arr2 = new Color32[arr.Length];
        int W = texture.width;
        int H = texture.height;
        int xc = W / 2;
        int yc = H / 2;

        for (j=0; j<H; j++) {
                for (i=0; i<W; i++) {
                        arr2 [j * W + i] = new Color32 (0, 0, 0, 0);
        
                        x = (int)(cs * (i - xc) + sn * (j - yc) + xc);
                        y = (int)(-sn * (i - xc) + cs * (j - yc) + yc);
        
                        if ((x > -1) && (x < W) && (y > -1) && (y < H)) { 
                                arr2 [j * W + i] = arr [y * W + x];
                        }
                }
        }

        Texture2D newImg = new Texture2D (W, H);
        newImg.SetPixels32 (arr2);
        newImg.Apply ();

        return newImg;
    }
}
