import gradio as gr
from utils import predict_news

# Interface configuration
def create_interface():
    with gr.Blocks(title="Fake News Detector", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("# 📰 Fake News Detector")
        gr.Markdown("**Classify news as REAL or FAKE using AI**")
        
        with gr.Row():
            with gr.Column():
                title_input = gr.Textbox(
                    label="News Title",
                    placeholder="Enter the title here...",
                    lines=2
                )
                
                text_input = gr.Textbox(
                    label="News Content", 
                    placeholder="Enter the full text here...",
                    lines=5
                )
                
                submit_btn = gr.Button("Check Authenticity", variant="primary")
            
            with gr.Column():
                result_label = gr.Label(label="Result")
                confidence = gr.Label(label="Model Confidence")
        
        # Predefined examples
        gr.Examples(
            examples=[
                ["Breaking: Scientists Discover New Planet", "Researchers at NASA have confirmed the discovery of a new exoplanet in the habitable zone of a distant star. The planet, named Kepler-452b, shows promising signs of potentially supporting life."],
                ["SHOCKING: Celebrity Secret Revealed", "You won't believe what this famous actor did behind closed doors! The truth will shock you and make you question everything about Hollywood!"]
            ],
            inputs=[title_input, text_input]
        )
        
        # Connect button to prediction function
        submit_btn.click(
            fn=predict_news,
            inputs=[title_input, text_input],
            outputs=[result_label, confidence]
        )
    
    return demo

# Launch the application
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)