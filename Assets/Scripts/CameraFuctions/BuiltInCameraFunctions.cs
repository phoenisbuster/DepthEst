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
    private WebCamDevice[] cam_devices;
    private int CameraNumber = 1;

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
    void OnDestroy() 
    {
        instance = null;
    }

    public void clickAccessCamera()
    {
        if(webcamTexture != null && !webcamTexture.isPlaying)
            accessCamera();
    }

    public void clickCapture()
    {
        StartCoroutine(SaveImage());
    } 

    public void changeCameraIndex()
    {
        if(CameraNumber <= 1)
        {
            return;
        }
        
        DefaultCameraIndex = DefaultCameraIndex == 0? 1 : 0;
        StopWebCamTex();
        initCamera();
    }

    public void changeImageRotate()
    {
        ImageScript.RotateImage();
    }

    public void PauseWebCamTex(bool isHide = false)
    {
        if(webcamTexture.isPlaying)
        {
            webcamTexture.Pause();
            if(isHide)
            {
                ImageScript.hideTexture();
            }
        }
    }

    public void StopWebCamTex()
    {
        if(webcamTexture.isPlaying)
        {
            webcamTexture.Stop();
            ImageScript.hideTexture();
        }
    }

    public void PlayWebCamTex()
    {
        if(!webcamTexture.isPlaying)
        {
            webcamTexture.Play();
            ImageScript.showTexture();
        }
    }

    private void initCamera()
    {
        //Obtain camera devices available
        cam_devices = WebCamTexture.devices;
        CameraNumber = cam_devices.Length;

        accessCamera();
    }

    private void accessCamera()
    {
        DebugLog.getInstance().updateLog(LogType.WebCamText, "Access Camera Start", false);
        
        //Set a camera to the webcamTexture
        Debug.Log("CHECK SCreen SIZE: " + Screen.width + " " + Screen.height);
        webcamTexture = new WebCamTexture(cam_devices[DefaultCameraIndex].name, 480, 480, 30);
        DebugLog.getInstance().updateLog(LogType.WebCamText, "Check Flip" + webcamTexture.videoVerticallyMirrored);
        DebugLog.getInstance().updateLog(LogType.WebCamText, "Check Rotate" + webcamTexture.videoRotationAngle);
        if(webcamTexture.videoVerticallyMirrored)
        {
            rawimage.GetComponent<RectTransform>().localScale = new Vector3(-1, 1, 1);
        }
        //Set the webcamTexture to the texture of the rawimage
        ImageScript.showTexture();
        ImageScript.ToggleHover(false);
        ImageScript.setTexture(webcamTexture);
        
        //Start the camera
        webcamTexture.Play();
    }

    private IEnumerator SaveImage()
    {
        DebugLog.getInstance().updateLog(LogType.WebCamText, "Save Pic Start", false);
        DebugLog.getInstance().updateLog(LogType.WebCamText, "CHECK IMAGE SIZE: " + rawimage.texture.width + " " + rawimage.texture.height);
        Debug.Log("CHECK IMAGE SIZE: " + rawimage.texture.width + " " + rawimage.texture.height);
        //Create a Texture2D with the size of the rendered image on the screen.
        Texture2D texture = new Texture2D(rawimage.texture.width, rawimage.texture.height, TextureFormat.ARGB32, false);
        //Save the image to the Texture2D
        texture.SetPixels(webcamTexture.GetPixels());
        //Rotate textture 90 degree to bypass the default option in Unity
        texture = RotateTexture(texture, -90);
        texture.Apply();

        webcamTexture.Pause();
        ImageScript.ToggleHover(true);

        Debug.Log("CHECK texture isReadable " + texture.isReadable);
        Debug.Log("CHECK rawimage isReadable " + rawimage.texture.isReadable);

        yield return new WaitForEndOfFrame();
        //StopWebCamTex();
        

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
        WSConnection.getInstance().setTextureData(texture.EncodeToPNG());
        RestFullAPI.TestAPI(texture.EncodeToPNG());
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
