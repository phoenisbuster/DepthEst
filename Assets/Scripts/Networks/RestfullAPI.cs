using System;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;
using UnityEngine;

class RestFullAPI
{
    public static async void TestAPI(byte[] content) 
    {
        await Main(content);
    }
    
    static async Task Main(byte[] content)
    {
        var httpClient = new HttpClient();
        var requestContent = new MultipartFormDataContent();
        var fileContent = new ByteArrayContent(content);
        fileContent.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("image/png");
        requestContent.Add(fileContent, "image", "image.png");
        var response = await httpClient.PostAsync("http://localhost:5000/upload", requestContent);
        var responseContent = await response.Content.ReadAsStringAsync();
        Debug.Log(responseContent);
    }
}
