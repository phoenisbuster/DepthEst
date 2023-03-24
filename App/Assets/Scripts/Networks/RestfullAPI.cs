using System;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;
using UnityEngine;

//"http://localhost:5000/upload"
public class RestFullAPI : MonoBehaviour
{   
    public static async void TestAPI(byte[] content) 
    {
        await Main(content);
    }
    
    static async Task Main(byte[] content)
    {
        try
        {    
            var httpClient = new HttpClient();
            var requestContent = new MultipartFormDataContent();
            var fileContent = new ByteArrayContent(content);
            fileContent.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("image/png");
            requestContent.Add(fileContent, "image", "image.png");

            var fullUrl = "http://" + WSConnection.getInstance().address + "/upload";
            Debug.Log(fullUrl);

            var response = await httpClient.PostAsync(fullUrl, requestContent);
            var responseContent = await response.Content.ReadAsStringAsync();
            Debug.Log(responseContent);
        }
        catch(Exception e)
        {
            Debug.LogWarning("Unable to connect to Server: " + e.Message);
        }
        finally
        {
            PanelDown.ToggleEditMode.Invoke(true, true);
        }
    }
}
