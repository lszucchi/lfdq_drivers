import numpy as np
import plotly.express as px
import os
from pathlib import Path
import re
from datetime import datetime
from IPython.display import clear_output


class Plot:
    def __init__(self, measure_name, measurement_info, dir_name, xaxis_name,
                 yaxis_name, name_info, title, data_x, data_y, data_z=None,
                 zaxis_name=None, width=800, height=500, bottom_margin=300):
        
        self.measure_name = measure_name
        self.measurement_info = measurement_info
        self.dir_name = dir_name
        self.xaxis_name = xaxis_name
        self.yaxis_name = yaxis_name
        self.zaxis_name = zaxis_name
        self.name_info = ', '.join(name_info)
        self.title = f"{title} - {self.name_info}"
        self.data_x = data_x
        self.data_y = data_y
        self.data_z = data_z
        self.width = width
        self.height = height
        self.bottom_margin = bottom_margin

    def _save_plot(self, fig, is_3d=False):
        """
        Helper function to save plot files only (HTML and PDF).
        Note: This method no longer saves measurement data to avoid duplication.
        Data saving is handled by the Measurement class's save_data() method.
        """
        # Ensure directory exists with absolute path to avoid path length issues
        try:
            # Create a more robust path handling
            output_dir_html = Path(self.dir_name) / "html"
            output_dir_pdf = Path(self.dir_name) / "pdf"
            
            output_dir_html.mkdir(parents=True, exist_ok=True)
            output_dir_pdf.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Use a shorter filename to avoid path length issues
            # Only include the first 30 characters of name_info to avoid excessively long paths
            safe_name_info = self.name_info.strip().lower()[:30]
            # Replace colons, spaces, and other problematic characters
            safe_name_info = safe_name_info.replace(":", "_").replace(" ", "_")
            safe_name_info = safe_name_info.replace(",", "_").replace("~", "to")
            # Replace any remaining invalid filename characters
            
            safe_name_info = re.sub(r'[^\w\-_.]', '_', safe_name_info)
            
            # Use a shorter base filename
            base_filename = f"{timestamp}_{self.measure_name}"
            if safe_name_info:
                base_filename += f"_{safe_name_info}"
            
            # Save plot files with absolute paths
            html_path = output_dir_html / f"{base_filename}.html"
            pdf_path = output_dir_pdf / f"{base_filename}.pdf"
            
            fig.write_html(str(html_path))
            fig.write_image(str(pdf_path), format="pdf")
            print(f"Plot saved to {html_path} and {pdf_path}")
            
        except Exception as e:
            print(f"Warning: Could not save plot files: {e}")
            # Try with a simpler filename if the original fails
            try:
                # Create directories again to be sure
                Path(self.dir_name).mkdir(parents=True, exist_ok=True)
                Path(self.dir_name, "html").mkdir(exist_ok=True)
                Path(self.dir_name, "pdf").mkdir(exist_ok=True)
                
                simple_base = f"{timestamp}_{self.measure_name}"
                html_path = Path(self.dir_name) / "html" / f"{simple_base}.html"
                pdf_path = Path(self.dir_name) / "pdf" / f"{simple_base}.pdf"
                
                fig.write_html(str(html_path))
                fig.write_image(str(pdf_path), format="pdf")
                print(f"Plot saved with simplified filename to {html_path}")
            except Exception as e2:
                print(f"Error: Could not save plot files with simplified name: {e2}")

    def plot(self, save=True):
        clear_output(wait=True)
        text_info = self.measurement_info.replace("\n", "<br>")

        fig = px.line(
            x=self.data_x,
            y=self.data_y,
            labels={'x': self.xaxis_name, 'y': self.yaxis_name}
        )
        
        fig.update_layout(
            title=self.title,
            xaxis_title=self.xaxis_name,
            yaxis_title=self.yaxis_name,
            width=self.width,
            height=self.height,
            annotations=[
                dict(
                    x=-0.1,
                    y=-0.3,
                    xref="paper",
                    yref="paper",
                    text=text_info,
                    showarrow=False,
                    font=dict(size=10),
                    align="left",
                    xanchor="left",
                    yanchor="top"
                )
            ],
            margin=dict(b=self.bottom_margin)
        )

        if save:
            self._save_plot(fig)

        return fig

    def plot_3d(self, save=True):
        clear_output(wait=True)
        text_info = self.measurement_info.replace("\n", "<br>")

        # Create heatmap using plotly express
        fig = px.imshow(
            self.data_z,
            labels={'x': self.xaxis_name, 'y': self.yaxis_name, 'color': self.zaxis_name},
            x=self.data_x,
            y=self.data_y,
            color_continuous_scale='Viridis',
            aspect='auto',
        )
        
        # Enhance the default figure settings for better visualization
        fig.update_layout(
            title=self.title,
            xaxis_title=self.xaxis_name,
            yaxis_title=self.yaxis_name,
            width=self.width,
            height=self.height,
            annotations=[
                dict(
                    x=-0.1,
                    y=-0.3,
                    xref="paper",
                    yref="paper",
                    text=text_info,
                    showarrow=False,
                    font=dict(size=10),
                    align="left",
                    xanchor="left",
                    yanchor="top"
                )
            ],
            margin=dict(b=300)
        )
        
        # Check if data contains NaN values and update colorscale accordingly
        if np.isnan(self.data_z).any():
            fig.update_traces(
                hovertemplate="X: %{x}<br>Y: %{y}<br>Value: %{z}<br><extra></extra>",
                # For heatmap, use a colorscale that can handle NaN values
                # Plotly automatically handles NaN values as gaps
                connectgaps=False
            )

        if save:
            self._save_plot(fig, is_3d=True)

        return fig