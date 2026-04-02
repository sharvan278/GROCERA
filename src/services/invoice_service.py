"""
Invoice Generation Service

Generates PDF invoices for orders using ReportLab.
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
from datetime import datetime
from typing import Optional
from src.models.multi_tenant import Order
import os


class InvoiceService:
    """Service for generating PDF invoices"""
    
    def __init__(self):
        self.invoice_dir = 'invoices'
        os.makedirs(self.invoice_dir, exist_ok=True)
    
    def generate_invoice(self, order: Order, output_path: Optional[str] = None) -> str:
        """
        Generate PDF invoice for an order.
        
        Args:
            order: Order object
            output_path: Optional custom output path
        
        Returns:
            Path to generated PDF file
        """
        if not output_path:
            output_path = os.path.join(
                self.invoice_dir,
                f"invoice_{order.order_number}.pdf"
            )
        
        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#4CAF50'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#333333'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        # Header - Company Info
        story.append(Paragraph("GROCERA", title_style))
        story.append(Paragraph("Inventory Management System", styles['Normal']))
        story.append(Paragraph("Email: support@grocera.com | Phone: +1 (555) 123-4567", styles['Normal']))
        story.append(Spacer(1, 0.3 * inch))
        
        # Invoice Title
        story.append(Paragraph(f"INVOICE #{order.order_number}", heading_style))
        story.append(Spacer(1, 0.2 * inch))
        
        # Invoice Details Table
        invoice_data = [
            ['Invoice Date:', order.placed_at.strftime('%B %d, %Y') if order.placed_at else datetime.utcnow().strftime('%B %d, %Y')],
            ['Order Number:', order.order_number],
            ['Payment Status:', order.payment_status.upper()],
            ['Payment Method:', order.payment_method.upper() if order.payment_method else 'N/A'],
        ]
        
        invoice_table = Table(invoice_data, colWidths=[2*inch, 3*inch])
        invoice_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#666666')),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#333333')),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica-Bold'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(invoice_table)
        story.append(Spacer(1, 0.3 * inch))
        
        # Customer Info
        story.append(Paragraph("BILL TO:", heading_style))
        customer_name = order.customer.username if order.customer else 'Customer'
        customer_email = order.customer.email if order.customer else 'N/A'
        story.append(Paragraph(f"Customer: {customer_name}", styles['Normal']))
        story.append(Paragraph(f"Email: {customer_email}", styles['Normal']))
        if order.delivery_address:
            story.append(Paragraph(f"Address: {order.delivery_address}", styles['Normal']))
        story.append(Spacer(1, 0.3 * inch))
        
        # Items Table
        story.append(Paragraph("ORDER ITEMS:", heading_style))
        
        # Table header
        items_data = [
            ['#', 'Item Name', 'Quantity', 'Unit Price', 'Subtotal']
        ]
        
        # Table rows
        for idx, item in enumerate(order.items, 1):
            items_data.append([
                str(idx),
                item.item_name,
                str(item.quantity),
                f"${item.unit_price:.2f}",
                f"${item.subtotal:.2f}"
            ])
        
        # Total row
        items_data.append([
            '', '', '', 'TOTAL:', f"${order.final_amount:.2f}"
        ])
        
        # Create table
        items_table = Table(items_data, colWidths=[0.5*inch, 3*inch, 1*inch, 1.2*inch, 1.3*inch])
        items_table.setStyle(TableStyle([
            # Header styling
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4CAF50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            
            # Body styling
            ('FONTNAME', (0, 1), (-1, -2), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -2), 10),
            ('ALIGN', (2, 1), (-1, -2), 'RIGHT'),
            ('ALIGN', (0, 1), (1, -2), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            
            # Grid
            ('GRID', (0, 0), (-1, -2), 0.5, colors.grey),
            ('LINEBELOW', (0, 0), (-1, 0), 2, colors.HexColor('#4CAF50')),
            
            # Total row styling
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, -1), (-1, -1), 12),
            ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#F5F5F5')),
            ('ALIGN', (0, -1), (-1, -1), 'RIGHT'),
            ('TEXTCOLOR', (0, -1), (-1, -1), colors.HexColor('#4CAF50')),
            ('LINEABOVE', (0, -1), (-1, -1), 2, colors.HexColor('#4CAF50')),
            
            # Padding
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(items_table)
        story.append(Spacer(1, 0.4 * inch))
        
        # Footer Notes
        if order.notes:
            story.append(Paragraph("NOTES:", heading_style))
            story.append(Paragraph(order.notes, styles['Normal']))
            story.append(Spacer(1, 0.2 * inch))
        
        # Thank you message
        story.append(Spacer(1, 0.3 * inch))
        story.append(Paragraph(
            "Thank you for your business!",
            ParagraphStyle(
                'ThankYou',
                parent=styles['Normal'],
                fontSize=12,
                alignment=TA_CENTER,
                textColor=colors.HexColor('#4CAF50'),
                spaceAfter=10
            )
        ))
        
        # Footer
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph(
            "This is a computer-generated invoice. No signature required.",
            ParagraphStyle(
                'Footer',
                parent=styles['Normal'],
                fontSize=8,
                alignment=TA_CENTER,
                textColor=colors.grey
            )
        ))
        
        # Build PDF
        doc.build(story)
        
        return output_path
    
    def generate_receipt(self, order: Order) -> str:
        """
        Generate a simple receipt (shorter than invoice).
        
        Args:
            order: Order object
        
        Returns:
            Path to generated PDF file
        """
        output_path = os.path.join(
            self.invoice_dir,
            f"receipt_{order.order_number}.pdf"
        )
        
        # Similar to invoice but simplified
        # Implementation can be added based on requirements
        return self.generate_invoice(order, output_path)


# Singleton instance
invoice_service = InvoiceService()
