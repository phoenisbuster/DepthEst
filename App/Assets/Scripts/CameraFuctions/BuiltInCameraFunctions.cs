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
        {
            accessCamera();
        } 
        else
        {
            Debug.LogWarning("Access Cam FAIL");
        }
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
        ImageLoader.changeStateSignal.Invoke(DefaultCameraIndex == 0? CameraState.BackCam : CameraState.FrontCam);
        StopWebCamTex();
        initCamera(true);
    }

    public void changeImageRotate()
    {
        ImageScript.RotateImage(true);
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

    private void initCamera(bool accessImidiate = false)
    {
        //Obtain camera devices available
        cam_devices = WebCamTexture.devices;
        CameraNumber = cam_devices.Length;

        if(accessImidiate)
            accessCamera();
    }

    public void accessCamera()
    {
        DebugLog.getInstance().updateLog(LogType.WebCamText, "Access Camera Start", false);
        
        //Set a camera to the webcamTexture
        Debug.Log("CHECK Screen SIZE: " + Screen.width + " " + Screen.height);
        webcamTexture = new WebCamTexture(cam_devices[DefaultCameraIndex].name, 480, 480, 30);
        Debug.Log(webcamTexture);
        DebugLog.getInstance().updateLog(LogType.WebCamText, "Check Flip" + webcamTexture.videoVerticallyMirrored);
        DebugLog.getInstance().updateLog(LogType.WebCamText, "Check Rotate" + webcamTexture.videoRotationAngle);

        //Set the webcamTexture to the texture of the rawimage
        ImageScript.showTexture();
        ImageScript.RotateImage(false, -90);
        ImageScript.changeCameraState(DefaultCameraIndex == 0? CameraState.BackCam : CameraState.FrontCam);
        ImageScript.setTexture(webcamTexture);
        
        StartCoroutine(changeImageSize());
        
        //Start the camera
        webcamTexture.Play();
    }

    private IEnumerator changeImageSize()
    {
        yield return new WaitForSeconds(0.5f);
        //ImageScript.changeImageSize(rawimage.texture.width, rawimage.texture.height);
        Debug.Log("Check access Cam Hei" + rawimage.texture.height);
        Debug.Log("Check access Cam Wid" + rawimage.texture.width);
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
        //ImageScript.ToggleHover(true);

        Debug.Log("CHECK texture isReadable " + texture.isReadable);
        Debug.Log("CHECK rawimage isReadable " + rawimage.texture.isReadable);
        byte[] bytes = texture.EncodeToPNG();
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
                                                                        RestAPI.TestAPI(bytes);
                                                                    }
        );
        SaveToResources.Save(texture.EncodeToPNG(), "TestWebCamText", "png");
        //WSConnection.getInstance().setTextureData(texture.EncodeToPNG());
        //RestFullAPI.TestAPI(texture.EncodeToPNG());
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
