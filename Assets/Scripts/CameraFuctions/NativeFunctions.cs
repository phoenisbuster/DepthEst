using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using UnityEngine.UI;

public class NativeFunctions : MonoBehaviour
{
    public RawImage rawimage;
    public Button recordBtn;
    public Button captureBtn;
    public Button changeCamera;
    public Button changeImgRotate;
    public int DefaultCameraIndex = 0;

    private static NativeFunctions instance;

    private void Awake() 
    {
        instance = this;
    }

    public static NativeFunctions getInstance()
    {
        return instance;
    }

    public void clickRecord()
    {
        if(checkCameraBusy())
        {
            return;
        }
        RecordVideo();
    }

    public void clickCapture()
    {
        if(checkCameraBusy())
        {
            return;
        }
        TakePicture(1024);
    } 

    public void changeCameraIndex()
    {
        DefaultCameraIndex = DefaultCameraIndex == 0? 1 : 0;
    }

    public void changeImageRotate()
    {
        Debug.Log(rawimage.transform.eulerAngles.z);
        var newZ = 0;
        if(rawimage.transform.eulerAngles.z == newZ)
        {
            newZ = -90;
        }
        rawimage.transform.eulerAngles = new Vector3(0, 0, newZ);
    }

    private bool checkCameraBusy()
    {
        return NativeCamera.IsCameraBusy();
    }

    private void TakePicture(int maxSize)
    {
        NativeCamera.Permission permission = NativeCamera.TakePicture(( path) =>
        {
            DebugLog.getInstance().updateLog(LogType.NativeCamera, "NativeCam Start: " + path, false);
            if( path != null )
            {
                // Create a Texture2D from the captured image
                Texture2D texture = NativeCamera.LoadImageAtPath( path, maxSize );
                if(texture == null)
                {
                    DebugLog.getInstance().updateLog(LogType.NativeCamera, "Couldn't load texture: " + path, false);
                    return;
                }

                // Asign texture to the RawImage in the Scene
                rawimage.texture = texture;
                rawimage.material.mainTexture = texture;
                // If a procedural texture is not destroyed manually, 
                // it will only be freed after a scene change
                // Save to gallery before Destroy
                StartCoroutine(SaveImage(texture));
                //Destroy(texture, 5f );
            }
        }, maxSize );

        DebugLog.getInstance().updateLog(LogType.NativeCamera, "NativeCam End with: " + permission);
    }

    private void RecordVideo()
    {
        NativeCamera.Permission permission = NativeCamera.RecordVideo( ( path ) =>
        {
            DebugLog.getInstance().updateLog(LogType.NativeCamera, "NativeVideo Start: file://" + path, false);
            if( path != null )
            {
                // Play the recorded video
                Handheld.PlayFullScreenMovie("file://" + path);
            }
        } );

        DebugLog.getInstance().updateLog(LogType.NativeCamera, "NativeVideo End with: " + permission);
    }

    private IEnumerator SaveImage(Texture2D texture)
    {
        Texture2D textureInstance = new Texture2D(rawimage.texture.width, rawimage.texture.height, TextureFormat.ARGB32, false);
        textureInstance = texture;

        yield return new WaitForEndOfFrame();

        NativeGallery.Permission permission = NativeGallery.SaveImageToGallery(
                                                                    textureInstance, 
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
        //SaveToResources.Save(textureInstance.EncodeToPNG(), "TestNativeCam", "png");
        Destroy(texture);
    }
}
