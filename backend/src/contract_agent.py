"""
AI Agent for automated contract monitoring and reporting.
Detects expiring contracts, conflicts, and generates daily reports.
"""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser

from src.config import Config
from src.database import DatabaseManager

logger = logging.getLogger(__name__)

class ContractAgent:
    """AI Agent for automated contract monitoring and reporting"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        
        # Initialize LLM for agent
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            api_key=Config.OPENAI_API_KEY
        )
        
        # Create prompt template for the agent
        self.agent_prompt = ChatPromptTemplate.from_template("""
        You are a contract management AI assistant. You help analyze contracts, detect conflicts, and generate reports.
        
        Available tools:
        - get_expiring_contracts: Get contracts expiring within specified days
        - find_conflicts: Find conflicts between contracts
        - get_contract_summary: Get summary of all active contracts
        - check_contract_status: Check status of specific contracts
        
        Question: {question}
        
        Provide a helpful and detailed response based on contract data.
        Use Markdown for formatting. For any math expressions, format using KaTeX-compatible LaTeX syntax (inline: $...$, block: $$...$$).
        """)
        
        logger.info("Contract monitoring agent initialized")
    
    
    def _get_expiring_contracts_tool(self, days: str) -> str:
        """Tool to get expiring contracts"""
        try:
            days_int = int(days) if days.isdigit() else Config.EXPIRATION_WARNING_DAYS
            contracts = self.db_manager.get_expiring_contracts(days_int)
            
            if not contracts:
                return f"No contracts expiring within {days_int} days."
            
            result = f"Found {len(contracts)} contract(s) expiring within {days_int} days:\n\n"
            for contract in contracts:
                result += f"- {contract.get('contract_name', 'Unnamed Contract')} "
                result += f"(File: {contract['documents']['filename']})\n"
                result += f"  End Date: {contract['end_date']}\n"
                result += f"  Parties: {', '.join(contract.get('parties', []))}\n"
                result += f"  Department: {contract.get('department', 'Unknown')}\n\n"
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting expiring contracts: {e}")
            return f"Error retrieving expiring contracts: {str(e)}"
    
    def _find_conflicts_tool(self, input_text: str) -> str:
        """Tool to find contract conflicts"""
        try:
            conflicts = self.db_manager.find_contract_conflicts()
            
            if not conflicts:
                return "No conflicts detected between contracts."
            
            result = f"Found {len(conflicts)} conflict(s):\n\n"
            
            for i, conflict in enumerate(conflicts, 1):
                result += f"Conflict #{i}:\n"
                result += f"Between: {conflict['contract1']['filename']} and {conflict['contract2']['filename']}\n"
                
                for conf_detail in conflict['conflicts']:
                    if conf_detail['type'] == 'contact_info_mismatch':
                        result += f"- Contact info mismatch in {conf_detail['field']}: "
                        result += f"'{conf_detail['value1']}' vs '{conf_detail['value2']}'\n"
                    elif conf_detail['type'] == 'end_date_mismatch':
                        result += f"- End date mismatch: {conf_detail['date1']} vs {conf_detail['date2']}\n"
                
                result += f"Detected: {conflict['detected_at']}\n\n"
            
            return result
            
        except Exception as e:
            logger.error(f"Error finding conflicts: {e}")
            return f"Error finding conflicts: {str(e)}"
    
    def _get_contract_summary_tool(self, input_text: str) -> str:
        """Tool to get contract summary"""
        try:
            # Get all active contracts
            result = self.db_manager.client.table('contracts')\
                .select('*, documents!inner(filename)')\
                .eq('status', 'active')\
                .execute()
            
            contracts = result.data
            
            if not contracts:
                return "No active contracts found in the system."
            
            summary = f"Active Contracts Summary ({len(contracts)} total):\n\n"
            
            # Group by department
            by_department = {}
            total_value = 0
            
            for contract in contracts:
                dept = contract.get('department', 'Unknown')
                if dept not in by_department:
                    by_department[dept] = []
                by_department[dept].append(contract)
            
            for dept, dept_contracts in by_department.items():
                summary += f"{dept} Department ({len(dept_contracts)} contracts):\n"
                for contract in dept_contracts:
                    summary += f"  - {contract.get('contract_name', 'Unnamed')}\n"
                    summary += f"    File: {contract['documents']['filename']}\n"
                    summary += f"    End Date: {contract.get('end_date', 'Unknown')}\n"
                summary += "\n"
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting contract summary: {e}")
            return f"Error retrieving contract summary: {str(e)}"
    
    def _check_contract_status_tool(self, query: str) -> str:
        """Tool to check specific contract status"""
        try:
            # Search for contracts by name or party
            result = self.db_manager.client.table('contracts')\
                .select('*, documents!inner(filename)')\
                .ilike('contract_name', f'%{query}%')\
                .execute()
            
            contracts = result.data
            
            if not contracts:
                # Try searching by parties
                all_contracts = self.db_manager.client.table('contracts')\
                    .select('*, documents!inner(filename)')\
                    .execute()
                
                contracts = [c for c in all_contracts.data 
                           if query.lower() in str(c.get('parties', [])).lower()]
            
            if not contracts:
                return f"No contracts found matching '{query}'"
            
            result_text = f"Found {len(contracts)} contract(s) matching '{query}':\n\n"
            
            for contract in contracts:
                result_text += f"Contract: {contract.get('contract_name', 'Unnamed')}\n"
                result_text += f"File: {contract['documents']['filename']}\n"
                result_text += f"Status: {contract.get('status', 'Unknown')}\n"
                result_text += f"Parties: {', '.join(contract.get('parties', []))}\n"
                result_text += f"Start Date: {contract.get('start_date', 'Unknown')}\n"
                result_text += f"End Date: {contract.get('end_date', 'Unknown')}\n"
                result_text += f"Department: {contract.get('department', 'Unknown')}\n\n"
            
            return result_text
            
        except Exception as e:
            logger.error(f"Error checking contract status: {e}")
            return f"Error checking contract status: {str(e)}"
    
    def generate_daily_report(self) -> Dict[str, Any]:
        """Generate comprehensive daily contract report"""
        try:
            logger.info("Generating daily contract report")
            
            # Get report data
            expiring_contracts = self.db_manager.get_expiring_contracts()
            conflicts = self.db_manager.find_contract_conflicts()
            
            # Use LLM to analyze and format the report
            analysis_prompt = ChatPromptTemplate.from_template("""
            As a contract management AI, analyze the following contract data and generate a comprehensive daily report:

            EXPIRING CONTRACTS:
            {expiring_data}

            CONFLICTS DETECTED:
            {conflicts_data}

            Please generate a professional daily report that includes:
            1. Executive Summary
            2. Urgent Actions Required (expiring contracts)
            3. Conflicts Detected (with specific details)
            4. Recommendations
            5. Next Steps

            Format the report professionally for email distribution using Markdown. For any math expressions, use KaTeX-compatible LaTeX syntax.
            """)
            
            chain = analysis_prompt | self.llm | StrOutputParser()
            
            response = chain.invoke({
                "expiring_data": json.dumps([{
                    'name': c.get('contract_name', 'Unnamed'),
                    'filename': c['documents']['filename'],
                    'end_date': c['end_date'],
                    'parties': c.get('parties', []),
                    'department': c.get('department')
                } for c in expiring_contracts], indent=2),
                "conflicts_data": json.dumps(conflicts, indent=2)
            })
            
            report_data = {
                "report_date": datetime.now().isoformat(),
                "expiring_contracts": len(expiring_contracts),
                "conflicts_found": len(conflicts),
                "report_content": response,
                "raw_data": {
                    "expiring": expiring_contracts,
                    "conflicts": conflicts
                }
            }
            
            logger.info(f"Daily report generated: {len(expiring_contracts)} expiring, {len(conflicts)} conflicts")
            
            return report_data
            
        except Exception as e:
            logger.error(f"Failed to generate daily report: {e}")
            return {
                "report_date": datetime.now().isoformat(),
                "error": str(e),
                "report_content": "Failed to generate daily report due to system error."
            }
    
    def send_email_report(self, report_data: Dict[str, Any], 
                         recipient_email: str = None) -> bool:
        """Send daily report via email"""
        try:
            recipient = recipient_email or Config.REPORT_EMAIL
            
            if not recipient:
                logger.error("No recipient email configured")
                return False
            
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = Config.EMAIL_USERNAME
            msg['To'] = recipient
            msg['Subject'] = f"Daily Contract Report - {datetime.now().strftime('%Y-%m-%d')}"
            
            # Email body
            body = f"""
Daily Contract Lifecycle Management Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{report_data.get('report_content', 'No content available')}

---
Automated report generated by CLM System
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(Config.SMTP_SERVER, Config.SMTP_PORT)
            server.starttls()
            server.login(Config.EMAIL_USERNAME, Config.EMAIL_PASSWORD)
            
            text = msg.as_string()
            server.sendmail(Config.EMAIL_USERNAME, recipient, text)
            server.quit()
            
            logger.info(f"Daily report email sent to {recipient}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email report: {e}")
            return False
    
    def run_daily_monitoring(self) -> Dict[str, Any]:
        """Run the complete daily monitoring cycle"""
        logger.info("Starting daily contract monitoring cycle")
        
        try:
            # Generate report
            report_data = self.generate_daily_report()
            
            # Send email if configured
            email_sent = False
            if Config.REPORT_EMAIL:
                email_sent = self.send_email_report(report_data)
            
            # Return summary
            return {
                "timestamp": datetime.now().isoformat(),
                "report_generated": True,
                "email_sent": email_sent,
                "expiring_contracts": report_data.get("expiring_contracts", 0),
                "conflicts_found": report_data.get("conflicts_found", 0),
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Daily monitoring failed: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "report_generated": False,
                "email_sent": False,
                "error": str(e),
                "status": "failed"
            }
    
    def query_agent(self, question: str) -> str:
        """Query the contract agent with a specific question"""
        try:
            # Determine what kind of query this is and get appropriate data
            if "expiring" in question.lower() or "expire" in question.lower():
                data = self._get_expiring_contracts_tool("30")
            elif "conflict" in question.lower():
                data = self._find_conflicts_tool("")
            elif "summary" in question.lower():
                data = self._get_contract_summary_tool("")
            else:
                # General query - get summary data
                data = self._get_contract_summary_tool("")
            
            # Use LLM to generate response
            prompt = ChatPromptTemplate.from_template("""
            You are a contract management AI assistant. Based on the contract data below, answer the user's question.
            
            Contract Data:
            {contract_data}
            
            User Question: {question}
            
            Provide a helpful and detailed response.
            Use Markdown formatting and KaTeX-compatible math when needed.
            """)
            
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({
                "contract_data": data,
                "question": question
            })
            
            return response
        except Exception as e:
            logger.error(f"Agent query failed: {e}")
            return f"I encountered an error while processing your question: {str(e)}"
